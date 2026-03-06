import asyncio
import aiofiles
import os
import json
import importlib.util
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import dotenv_values

# Configure logging
logger = logging.getLogger(__name__)

# 定义插件目录和清单文件名
PLUGIN_DIR = Path(__file__).parent
MANIFEST_FILE_NAME = "plugin-manifest.json"

class PluginManager:
    def __init__(self):
        # 存储插件 manifest
        self.plugins = {}
        # 存储预处理器模块
        self.message_preprocessors = {}
        # 存储服务模块
        self.service_modules = {}
        # 存储工具别名映射: "DailyNote.create" -> ("DailyNote", {"command": "create"})
        self.tool_aliases = {}
        self.web_socket_server = None
        self.vector_db_manager = None
        self.project_base_path = None

    # ========== 依赖注入 ==========
    def set_web_socket_server(self, wss):
        self.web_socket_server = wss

    def set_vector_db_manager(self, vdb_manager):
        self.vector_db_manager = vdb_manager

    def set_project_base_path(self, base_path):
        self.project_base_path = base_path

    # ========== 配置获取 ==========
    def _get_plugin_config(self, plugin_manifest):
        config = {}
        # 获取系统环境变量
        global_env = os.environ
        # 获取插件特定环境配置
        plugin_specific_env = plugin_manifest.get("pluginSpecificEnvConfig", {})

        if "configSchema" in plugin_manifest:
            for key, schema_entry in plugin_manifest["configSchema"].items():
                # 确定期望的类型
                if isinstance(schema_entry, dict) and "type" in schema_entry:
                    expected_type = schema_entry["type"]
                else:
                    expected_type = schema_entry

                # 获取原始值
                raw_value = None
                if key in plugin_specific_env:
                    raw_value = plugin_specific_env[key]
                elif key in global_env:
                    raw_value = global_env[key]
                else:
                    continue

                # 类型转换
                if expected_type == "integer":
                    config[key] = int(raw_value)
                elif expected_type == "boolean":
                    config[key] = str(raw_value).lower() == "true"
                else:
                    config[key] = raw_value

        # 处理 DebugMode 配置
        debug_mode = plugin_specific_env.get("DebugMode") or global_env.get("DebugMode") or "false"
        config["DebugMode"] = debug_mode.lower() == "true"
        return config

    # ========== 核心方法：调用预处理器 ==========
    async def execute_message_preprocessor(self, plugin_name, messages):
        processor_module = self.message_preprocessors.get(plugin_name)
        plugin_manifest = self.plugins.get(plugin_name)

        if not processor_module or not plugin_manifest:
            print(f"[PluginManager] Plugin \"{plugin_name}\" not found.")
            return messages

        if not hasattr(processor_module, "process_messages") or not callable(getattr(processor_module, "process_messages")):
            print(f"[PluginManager] Plugin \"{plugin_name}\" does not have 'process_messages' function.")
            return messages

        try:
            plugin_config = self._get_plugin_config(plugin_manifest)
            # 调用异步/同步的 process_messages 方法
            process_func = getattr(processor_module, "process_messages")
            if asyncio.iscoroutinefunction(process_func):
                processed_messages = await process_func(messages, plugin_config)
            else:
                processed_messages = process_func(messages, plugin_config)
            return processed_messages
        except Exception as error:
            print(f"[PluginManager] Error in message preprocessor {plugin_name}:", error)
            return messages

    # ========== 核心方法：处理工具调用 ==========
    async def process_tool_call(self, tool_name: str, tool_args: dict) -> dict:
        """
        处理工具调用 - 根据协议类型路由到对应的插件

        Args:
            tool_name: 工具/插件名称
            tool_args: 工具参数

        Returns:
            插件执行结果
        """
        # 解析别名
        if tool_name in self.tool_aliases:
            real_plugin_name, additional_args = self.tool_aliases[tool_name]
            # 合并别名参数和用户提供的参数（用户参数优先）
            tool_args = {**additional_args, **tool_args}
            tool_name = real_plugin_name
            logger.debug(f"[PluginManager] Resolved alias '{tool_name}' -> '{real_plugin_name}' with args: {additional_args}")

        plugin = self.plugins.get(tool_name)

        if not plugin:
            logger.error(f"[PluginManager] Plugin not found: {tool_name}")
            print(f"[PluginManager] Plugin \"{tool_name}\" not found.")
            return {"status": "error", "error": f"Plugin '{tool_name}' not found"}

        protocol = plugin.get("communication", {}).get("protocol", "direct")

        if protocol == "stdio":
            return await self._execute_stdio_plugin(tool_name, tool_args)
        elif protocol == "direct":
            module = self.get_service_module(tool_name)
            if module and hasattr(module, "process_tool_call"):
                if asyncio.iscoroutinefunction(module.process_tool_call):
                    result = await module.process_tool_call(tool_args)
                else:
                    result = module.process_tool_call(tool_args)
                return result
            else:
                logger.error(f"[PluginManager] Plugin '{tool_name}' does not support tool calls")
                return {"status": "error", "error": f"Plugin '{tool_name}' does not support tool calls"}
        else:
            return {"status": "error", "error": f"Unknown protocol: {protocol}"}

    # ========== stdio 协议插件执行 ==========
    async def _execute_stdio_plugin(self, plugin_name: str, input_data: dict) -> dict:
        """
        执行 stdio 协议插件 (如 DeepMemo)

        通过子进程执行外部可执行文件，通过 stdin 传递 JSON 输入，
        通过 stdout 读取 JSON 输出。
        """
        plugin = self.plugins.get(plugin_name)
        if not plugin:
            return {"status": "error", "error": f"Plugin '{plugin_name}' not found"}

        entry_point = plugin.get("entryPoint", {})
        command = entry_point.get("command", "")
        timeout = plugin.get("communication", {}).get("timeout", 60000) / 1000  # 转换为秒

        base_path = plugin.get("basePath", "")

        if not command:
            return {"status": "error", "error": f"Plugin '{plugin_name}' has no command configured"}

        try:
            # 创建子进程
            process = await asyncio.create_subprocess_exec(
                *command.split(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=base_path
            )

            # 准备输入 JSON
            input_json = json.dumps(input_data, ensure_ascii=False)

            # 执行并等待结果
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_json.encode('utf-8')),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"[PluginManager] {plugin_name} timeout")
                process.kill()
                await process.wait()
                return {"status": "error", "error": f"Plugin '{plugin_name}' execution timeout"}

            # 检查返回码
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='replace') if stderr else "Unknown error"
                logger.error(f"[PluginManager] {plugin_name} failed (code {process.returncode}): {error_msg}")
                return {"status": "error", "error": f"Plugin '{plugin_name}' failed: {error_msg}"}

            # 解析输出
            try:
                stdout_str = stdout.decode('utf-8')
                result = json.loads(stdout_str)
                return result
            except json.JSONDecodeError as e:
                logger.error(f"[PluginManager] Invalid JSON from {plugin_name}: {e}")
                return {"status": "error", "error": f"Invalid JSON output from plugin: {e}"}

        except FileNotFoundError:
            return {"status": "error", "error": f"Command not found: {command}"}
        except Exception as e:
            return {"status": "error", "error": f"Failed to execute plugin: {str(e)}"}

    # ========== 插件加载 ==========
    async def load_plugins(self):
        """加载所有插件 (deepmemo, rag_daily, daily_note)"""
        logger.info("[PluginManager] Loading all plugins...")
        await self._load_deepmemo_plugin()
        await self._load_rag_daily_plugin()
        await self._load_daily_note_plugin()
        logger.info(f"[PluginManager] Plugin loading complete. Loaded plugins: {list(self.plugins.keys())}")

    async def _load_deepmemo_plugin(self):
        """加载 stdio 协议的 DeepMemo 插件"""
        logger.info("[PluginManager] Loading DeepMemo plugin...")
        plugin_path = PLUGIN_DIR / "deepmemo"
        manifest_path = plugin_path / MANIFEST_FILE_NAME

        try:
            # 1. 读取 manifest
            async with aiofiles.open(manifest_path, "r", encoding="utf-8") as f:
                manifest_content = await f.read()
            manifest = json.loads(manifest_content)
            manifest["basePath"] = str(plugin_path)

            # 2. 读取 config.env（如果存在）
            try:
                env_path = plugin_path / "config.env"
                if env_path.exists():
                    manifest["pluginSpecificEnvConfig"] = dotenv_values(env_path)
                    logger.info(f"[PluginManager] Loaded config.env from {env_path}")
                else:
                    manifest["pluginSpecificEnvConfig"] = {}
                    logger.info(f"[PluginManager] No config.env found at {env_path}")
            except Exception as e:
                manifest["pluginSpecificEnvConfig"] = {}
                logger.warning(f"[PluginManager] Error loading config.env: {e}")

            # 3. 保存 manifest (stdio 协议不需要加载模块)
            self.plugins[manifest["name"]] = manifest

            print(f"[PluginManager] Loaded: {manifest['displayName']} ({manifest['name']}) - stdio protocol")
            logger.info(f"[PluginManager] Successfully loaded: {manifest['displayName']} ({manifest['name']}) - stdio protocol")

        except FileNotFoundError:
            print(f"[PluginManager] DeepMemo plugin manifest not found, skipping...")
            logger.warning(f"[PluginManager] DeepMemo plugin manifest not found at {manifest_path}")
        except Exception as error:
            print(f"[PluginManager] Error loading DeepMemo plugin:", error)
            logger.error(f"[PluginManager] Error loading DeepMemo plugin: {error}", exc_info=True)

    async def _load_rag_daily_plugin(self):
        """加载 direct 协议的 RAGDailyPlugin"""
        logger.info("[PluginManager] Loading RAGDailyPlugin...")
        plugin_path = PLUGIN_DIR / "rag_daily"
        manifest_path = plugin_path / MANIFEST_FILE_NAME

        try:
            # 1. 读取 manifest
            async with aiofiles.open(manifest_path, "r", encoding="utf-8") as f:
                manifest_content = await f.read()
            manifest = json.loads(manifest_content)
            manifest["basePath"] = str(plugin_path)

            # 2. 读取 config.env（如果存在）
            try:
                env_path = plugin_path / "config.env"
                if env_path.exists():
                    manifest["pluginSpecificEnvConfig"] = dotenv_values(env_path)
                    logger.info(f"[PluginManager] Loaded config.env from {env_path}")
                else:
                    manifest["pluginSpecificEnvConfig"] = {}
                    logger.info(f"[PluginManager] No config.env found at {env_path}")
            except Exception as e:
                manifest["pluginSpecificEnvConfig"] = {}
                logger.warning(f"[PluginManager] Error loading config.env: {e}")

            # 3. 动态加载模块
            entry_point_script = manifest["entryPoint"]["script"]
            script_path = plugin_path / entry_point_script

            # 确保脚本路径存在
            if not script_path.exists():
                raise FileNotFoundError(f"Entry point script not found: {script_path}")

            # Python 动态导入模块
            spec = importlib.util.spec_from_file_location(
                f"plugin_{manifest['name']}",
                str(script_path)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info(f"[PluginManager] Loaded module from {script_path}")

            # 4. 注册为预处理器
            if hasattr(module, "process_messages") and callable(getattr(module, "process_messages")):
                self.message_preprocessors[manifest["name"]] = module
                logger.info(f"[PluginManager] Registered {manifest['name']} as message preprocessor")

            # 5. 注册为服务
            self.service_modules[manifest["name"]] = {
                "manifest": manifest,
                "module": module
            }

            # 6. 保存 manifest
            self.plugins[manifest["name"]] = manifest

            print(f"[PluginManager] Loaded: {manifest['displayName']} ({manifest['name']}) - direct protocol")
            logger.info(f"[PluginManager] Successfully loaded: {manifest['displayName']} ({manifest['name']}) - direct protocol")

            # 7. 初始化插件
            if hasattr(module, "initialize") and callable(getattr(module, "initialize")):
                config = self._get_plugin_config(manifest)
                config["PORT"] = os.environ.get("PORT")
                config["Key"] = os.environ.get("Key")
                config["PROJECT_BASE_PATH"] = self.project_base_path

                dependencies = {
                    "vectorDBManager": self.vector_db_manager,
                }

                logger.info(f"[PluginManager] Initializing {manifest['name']} with config...")
                # 处理异步/同步初始化
                init_func = getattr(module, "initialize")
                if asyncio.iscoroutinefunction(init_func):
                    await init_func(config, dependencies)
                else:
                    init_func(config, dependencies)
                logger.info(f"[PluginManager] {manifest['name']} initialized successfully")

        except FileNotFoundError:
            print(f"[PluginManager] RAGDailyPlugin manifest not found, skipping...")
            logger.warning(f"[PluginManager] RAGDailyPlugin manifest not found at {manifest_path}")
        except Exception as error:
            print(f"[PluginManager] Error loading RAGDailyPlugin:", error)
            logger.error(f"[PluginManager] Error loading RAGDailyPlugin: {error}", exc_info=True)

    async def _load_daily_note_plugin(self):
        """加载 stdio 协议的 DailyNote 插件"""
        logger.info("[PluginManager] Loading DailyNote plugin...")
        plugin_path = PLUGIN_DIR / "daily_note"
        manifest_path = plugin_path / MANIFEST_FILE_NAME

        try:
            # 1. 读取 manifest
            async with aiofiles.open(manifest_path, "r", encoding="utf-8") as f:
                manifest_content = await f.read()
            manifest = json.loads(manifest_content)
            manifest["basePath"] = str(plugin_path)

            # 2. 读取 config.env（如果存在）
            try:
                env_path = plugin_path / "config.env"
                if env_path.exists():
                    manifest["pluginSpecificEnvConfig"] = dotenv_values(env_path)
                    logger.info(f"[PluginManager] Loaded config.env from {env_path}")
                else:
                    manifest["pluginSpecificEnvConfig"] = {}
                    logger.info(f"[PluginManager] No config.env found at {env_path}")
            except Exception as e:
                manifest["pluginSpecificEnvConfig"] = {}
                logger.warning(f"[PluginManager] Error loading config.env: {e}")

            # 3. 保存 manifest (stdio 协议不需要加载模块)
            self.plugins[manifest["name"]] = manifest

            # 4. 处理工具别名（从 invocationCommands 生成）
            capabilities = manifest.get("capabilities", {})
            invocation_commands = capabilities.get("invocationCommands", [])
            plugin_name = manifest["name"]

            for cmd in invocation_commands:
                cmd_name = cmd.get("command", "")
                # 如果命令名包含 "."，说明是子命令，需要注册别名
                if "." in cmd_name:
                    # 提取子命令部分，例如 "DailyNote.create" -> "create"
                    subcommand = cmd_name.split(".")[-1]
                    # 解析 example 中的 command 参数
                    example = cmd.get("example", "")
                    additional_args = {"command": subcommand}

                    # 注册别名映射
                    self.tool_aliases[cmd_name] = (plugin_name, additional_args)
                    logger.info(f"[PluginManager] Registered tool alias: '{cmd_name}' -> '{plugin_name}' with command='{subcommand}'")

            print(f"[PluginManager] Loaded: {manifest['displayName']} ({manifest['name']}) - stdio protocol")
            logger.info(f"[PluginManager] Successfully loaded: {manifest['displayName']} ({manifest['name']}) - stdio protocol")

        except FileNotFoundError:
            print(f"[PluginManager] DailyNote plugin manifest not found, skipping...")
            logger.warning(f"[PluginManager] DailyNote plugin manifest not found at {manifest_path}")
        except Exception as error:
            print(f"[PluginManager] Error loading DailyNote plugin:", error)
            logger.error(f"[PluginManager] Error loading DailyNote plugin: {error}", exc_info=True)

    # ========== 关闭 ==========
    async def shutdown_all_plugins(self):
        for name, service_data in self.service_modules.items():
            module = service_data["module"]
            if module and hasattr(module, "shutdown") and callable(getattr(module, "shutdown")):
                try:
                    shutdown_func = getattr(module, "shutdown")
                    if asyncio.iscoroutinefunction(shutdown_func):
                        await shutdown_func()
                    else:
                        shutdown_func()
                except Exception as error:
                    print(f"[PluginManager] Error shutting down {name}:", error)

        if self.vector_db_manager and hasattr(self.vector_db_manager, "shutdown") and callable(getattr(self.vector_db_manager, "shutdown")):
            shutdown_func = getattr(self.vector_db_manager, "shutdown")
            if asyncio.iscoroutinefunction(shutdown_func):
                await shutdown_func()
            else:
                shutdown_func()

    # ========== 工具描述获取 ==========
    def get_tool_descriptions(self) -> Dict[str, str]:
        """
        Get tool descriptions for all plugins.
        Used for injecting into system prompt.

        Returns:
            {
                'VCPDeepMemo': 'Tool description text...',
                'VCPPluginName': 'Tool description text...'
            }
        """
        descriptions = {}

        for name, manifest in self.plugins.items():
            capabilities = manifest.get('capabilities', {})
            invocation_commands = capabilities.get('invocationCommands', [])

            if invocation_commands:
                # Build tool description
                for cmd in invocation_commands:
                    cmd_name = cmd.get('command', name)
                    cmd_desc = cmd.get('description', '')
                    cmd_example = cmd.get('example', '')

                    description = f"""## {cmd_name}

{cmd_desc}

示例参数:
```json
{cmd_example}
```

当需要使用此工具时，请按以下格式输出：

<<<[TOOL_REQUEST]>>>
tool_name:「始」{cmd_name}「末」,
参数名1:「始」值1「末」,
参数名2:「始」值2「末」
<<<[END_TOOL_REQUEST]>>>

请确保：
1. 所有参数使用「始」和「末」包裹
2. 参数之间用逗号分隔
3. 工具调用用 <<<[TOOL_REQUEST]>>> 包裹
"""
                    descriptions[f'VCP{cmd_name}'] = description

        return descriptions

    def get_all_tools_description(self) -> str:
        """
        Get aggregated description of all available tools.

        Returns:
            Formatted description string for system prompt
        """
        descriptions = self.get_tool_descriptions()
        if not descriptions:
            return ""

        return """# 可用工具

你有权访问以下工具。当用户请求需要使用工具时，请使用相应的工具：

""" + "\n\n".join(descriptions.values())

    # ========== 获取器 ==========
    def get_plugin(self, name):
        return self.plugins.get(name)

    def get_service_module(self, name):
        service_data = self.service_modules.get(name)
        return service_data["module"] if service_data else None

# 创建单例实例
plugin_manager = PluginManager()

# 导出单例
__all__ = ["plugin_manager", "PluginManager"]