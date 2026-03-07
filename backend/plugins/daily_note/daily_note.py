"""DailyNote Plugin - 日记创建与更新插件 (stdio 协议)

从 stdin 读取 JSON 输入，处理后输出 JSON 到 stdout。

支持的命令：
- create: 创建新日记
- update: 更新现有日记
"""

import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime


# --- Configuration ---
# 默认角色目录和日记目录
DEFAULT_CHARACTERS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "characters"
DEFAULT_DAILY_DIR = Path(__file__).parent.parent.parent.parent / "data" / "daily"


def get_characters_dir() -> Path:
    """获取角色目录"""
    path_str = os.getenv("CHARACTERS_DIR")
    if path_str:
        return Path(path_str)
    return DEFAULT_CHARACTERS_DIR


def get_daily_dir() -> Path:
    """获取全局日记目录"""
    return DEFAULT_DAILY_DIR


def get_character_metadata(character_id: str, characters_dir: Path) -> dict:
    """根据 character_id (UUID) 获取角色元数据"""
    character_dir = characters_dir / character_id
    meta_file = character_dir / ".character_meta.json"

    if meta_file.exists():
        try:
            return json.loads(meta_file.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, IOError):
            pass

    return None


def sanitize_path_component(name):
    """路径组件清理"""
    if not name or not isinstance(name, str):
        return 'Untitled'

    sanitized = name
    # 移除路径分隔符和非法字符
    sanitized = re.sub(r'[\\/:*?"<>|]', '', sanitized)
    # 移除控制字符
    sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)
    # 将空白字符替换为下划线
    sanitized = re.sub(r'\s+', '_', sanitized)
    # 移除开头和结尾的点和下划线
    sanitized = re.sub(r'^[._]+|[._]+$', '', sanitized)
    # 合并多个连续的下划线
    sanitized = re.sub(r'_+', '_', sanitized)

    # 长度限制
    if len(sanitized) > 100:
        sanitized = sanitized[:100].rstrip('._')

    return sanitized or 'Untitled'


def is_path_within_base(target_path, base_path):
    """路径安全验证"""
    resolved_target = Path(target_path).resolve()
    resolved_base = Path(base_path).resolve()
    return str(resolved_target) == str(resolved_base) or str(resolved_target).startswith(str(resolved_base) + os.sep)


def detect_tag_line(content):
    """检测内容中是否有 Tag 行"""
    lines = content.split('\n')
    if not lines:
        return {'has_tag': False, 'last_line': '', 'content_without_last_line': content}

    last_line = lines[-1].strip()
    tag_pattern = re.compile(r'^Tag:\s*.+', re.IGNORECASE)
    has_tag = tag_pattern.match(last_line) is not None
    content_without_last_line = '\n'.join(lines[:-1]) if has_tag else content

    return {'has_tag': has_tag, 'last_line': last_line, 'content_without_last_line': content_without_last_line}


def fix_tag_format(tag_line):
    """修复 Tag 格式"""
    fixed = tag_line.strip()
    fixed = re.sub(r'^tag:\s*', 'Tag: ', fixed, flags=re.IGNORECASE)
    if not fixed.startswith('Tag: '):
        fixed = 'Tag: ' + fixed

    tag_content = fixed[5:].strip()
    # 标准化标签内容
    normalized_content = tag_content
    normalized_content = re.sub(r'[\uff1a]', '', normalized_content)  # 全中文冒号
    normalized_content = re.sub(r'[\uff0c]', ', ', normalized_content)  # 全中文逗号
    normalized_content = re.sub(r'[\u3001]', ', ', normalized_content)  # 中文顿号
    normalized_content = re.sub(r'[。.]+$', '', normalized_content)  # 移除末尾句号
    normalized_content = re.sub(r',\s*', ', ', normalized_content)
    normalized_content = re.sub(r',\s{2,}', ', ', normalized_content)
    normalized_content = re.sub(r'\s+,', ',', normalized_content)
    normalized_content = re.sub(r'\s{2,}', ' ', normalized_content).strip()

    return 'Tag: ' + normalized_content


def process_tags(content_text, external_tag=None):
    """处理 Tag"""
    # 优先使用外部 Tag
    if external_tag and isinstance(external_tag, str) and external_tag.strip():
        fixed_tag = fix_tag_format(external_tag)
        return content_text.rstrip() + '\n' + fixed_tag

    # 从内容中检测 Tag
    detection = detect_tag_line(content_text)

    if detection['has_tag']:
        fixed_tag = fix_tag_format(detection['last_line'])
        return detection['content_without_last_line'].rstrip() + '\n' + fixed_tag
    else:
        raise ValueError("Tag is missing. Please provide a 'Tag' argument or add a 'Tag:' line at the end of the 'Content'.")


def handle_create(args):
    """处理 create 命令"""
    # 兼容不同的参数名
    maid = args.get('maid') or args.get('maidName') or args.get('Maid') or args.get('MAID')
    date_string = args.get('dateString') or args.get('Date')
    content_text = args.get('contentText') or args.get('Content')
    tag = args.get('Tag') or args.get('tag')

    if not maid or not date_string or not content_text:
        return {
            "status": "error",
            "error": "Invalid input for create: Missing maid/maidName, dateString/Date, or contentText/Content."
        }

    try:
        processed_content = process_tags(content_text, tag)

        # 解析 maid: 支持 [folder]author 格式，但使用 character_id 作为目录
        actual_maid_name = maid.strip()
        tag_match = re.match(r'^\[(.*?)\](.*)$', actual_maid_name)
        if tag_match:
            actual_maid_name = tag_match.group(2).strip()

        # character_id 就是 maid（去除 [folder] 前缀后的值）
        character_id = actual_maid_name

        # 清理日期格式
        date_part = date_string.replace('.', '-').replace('\\', '-').replace('/', '-').strip()
        date_part = re.sub(r'-+', '-', date_part)

        now = datetime.now()
        time_string = now.strftime("%H_%M_%S")

        characters_dir = get_characters_dir()
        daily_dir = get_daily_dir()

        # 从 .character_meta.json 获取角色名称
        character_meta = get_character_metadata(character_id, characters_dir)
        character_name = character_meta.get("name", character_id)

        sanitized_name = sanitize_path_component(character_name)

        # 日记目录: data/daily/{name}/
        name_daily_dir = daily_dir / sanitized_name

        # 安全检查
        if not is_path_within_base(name_daily_dir, daily_dir):
            return {
                "status": "error",
                "error": "Security error: Invalid folder path detected."
            }

        # 创建目录
        name_daily_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        base_file_name = f"{date_part}-{time_string}"
        file_extension = ".txt"
        final_file_name = f"{base_file_name}{file_extension}"
        file_path = name_daily_dir / final_file_name
        counter = 1

        # 循环检查文件名冲突
        while file_path.exists():
            counter += 1
            final_file_name = f"{base_file_name}({counter}){file_extension}"
            file_path = name_daily_dir / final_file_name

        # 写入文件
        file_content = f"[{date_part}] - {character_name}\n{processed_content}"
        file_path.write_text(file_content, encoding='utf-8')

        # 计算相对路径 (格式: {name}/{filename}.txt)
        relative_path = file_path.relative_to(daily_dir).as_posix()

        return {
            "status": "success",
            "message": f"Diary saved to {relative_path}",
            "path": relative_path,
            "character_id": character_id,
            "character_name": character_name,
            "date": date_part
        }

    except ValueError as e:
        return {
            "status": "error",
            "error": str(e)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e) or "An unknown error occurred during diary creation."
        }


def handle_update(args):
    """处理 update 命令"""
    target = args.get('target')
    replace = args.get('replace')
    maid = args.get('maid')

    if not isinstance(target, str) or not isinstance(replace, str):
        return {
            "status": "error",
            "error": "Invalid arguments for update: 'target' and 'replace' must be strings."
        }

    if len(target) < 15:
        return {
            "status": "error",
            "error": f"Security check failed: 'target' must be at least 15 characters long. Provided length: {len(target)}"
        }

    try:
        daily_dir = get_daily_dir()
        characters_dir = get_characters_dir()
        modification_done = False
        modified_file_path = None

        # 构建搜索顺序
        priority_names = []
        other_names = []

        # 获取所有日记目录
        if not daily_dir.exists():
            return {
                "status": "error",
                "error": f"Daily directory not found at {daily_dir}"
            }

        all_daily_dirs = [
            d for d in daily_dir.iterdir()
            if d.is_dir()
        ]

        if maid:
            # 解析 maid 获取 character_id
            character_id = maid.strip()
            tag_match = re.match(r'^\[(.*?)\](.*)$', character_id)
            if tag_match:
                character_id = tag_match.group(2).strip()

            # 获取角色名称
            character_meta = get_character_metadata(character_id, characters_dir)
            character_name = character_meta.get("name", character_id)
            sanitized_name = sanitize_path_component(character_name)

            # 优先搜索指定的角色名称目录
            for dir_entry in all_daily_dirs:
                if dir_entry.name == sanitized_name:
                    priority_names.append(dir_entry)
                else:
                    other_names.append(dir_entry)
        else:
            # 搜索所有目录
            other_names = all_daily_dirs

        # 合并搜索顺序：优先目录在前
        directories_to_scan = priority_names + other_names

        if not directories_to_scan:
            return {
                "status": "error",
                "error": f"No diary directories found in {daily_dir}"
            }

        for name_daily_dir in directories_to_scan:
            if modification_done:
                break

            if not name_daily_dir.exists():
                continue

            try:
                files = list(name_daily_dir.iterdir())
                txt_files = sorted([
                    f for f in files
                    if f.is_file() and f.suffix.lower() == '.txt'
                ], key=lambda x: x.name)

                for file_path in txt_files:
                    if modification_done:
                        break

                    try:
                        content = file_path.read_text(encoding='utf-8')
                    except Exception:
                        continue

                    index = content.find(target)
                    if index != -1:
                        new_content = content[:index] + replace + content[index + len(target):]
                        try:
                            file_path.write_text(new_content, encoding='utf-8')
                            modification_done = True
                            modified_file_path = file_path.relative_to(daily_dir).as_posix()
                            break
                        except Exception:
                            break
            except Exception:
                continue

        if modification_done:
            return {
                "status": "success",
                "message": f"Successfully edited diary file: {modified_file_path}",
                "path": modified_file_path
            }
        else:
            error_message = f"Target content not found in any diary files for maid '{maid}'." if maid else "Target content not found in any diary files."
            return {
                "status": "error",
                "error": error_message
            }

    except Exception as error:
        return {
            "status": "error",
            "error": f"An unexpected error occurred: {str(error)}"
        }


def main():
    """主函数：读取 stdin，处理命令，输出结果到 stdout"""
    try:
        # 读取 stdin
        input_data = sys.stdin.read()

        if not input_data:
            result = {
                "status": "error",
                "error": "No input data received via stdin."
            }
        else:
            args = json.loads(input_data)
            command = args.get('command', '')

            if command == 'create':
                params = {k: v for k, v in args.items() if k != 'command'}
                result = handle_create(params)
            elif command == 'update':
                params = {k: v for k, v in args.items() if k != 'command'}
                result = handle_update(params)
            else:
                result = {
                    "status": "error",
                    "error": f"Unknown command: '{command}'. Use 'create' or 'update'."
                }

        # 输出结果到 stdout
        sys.stdout.write(json.dumps(result, ensure_ascii=False))

        if result.get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)

    except json.JSONDecodeError as e:
        sys.stdout.write(json.dumps({
            "status": "error",
            "error": str(e) or "An unknown error occurred."
        }, ensure_ascii=False))
        sys.exit(1)
    except Exception as e:
        sys.stdout.write(json.dumps({
            "status": "error",
            "error": str(e) or "An unknown error occurred."
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
