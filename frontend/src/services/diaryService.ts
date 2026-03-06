/** Diary service for managing character diary entries

Diaries are stored in data/characters/{character_id}/daily/
*/

export interface DiaryEntry {
  path: string;           // 文件相对路径，例如 "{uuid}/daily/2025-01-23-14_30_52.txt"
  character_id: string;   // 角色 ID
  content: string;        // 日记内容（包含末尾的 Tag 行）
  mtime: number;          // 文件修改时间戳
}

/** Diary group by month for timeline display */
export interface DiaryGroup {
  year: number;
  month: number;
  monthLabel: string; // "2026年1月"
  diaries: DiaryEntry[];
  count: number;
  expanded: boolean; // UI state: whether this group is expanded
}

/** Map key format: "year-month" e.g., "2026-1" */
export type DiaryGroupMap = Record<string, DiaryEntry[]>;

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

/**
 * 获取日记列表
 * @param characterId - 角色 ID
 * @param limit - 返回数量限制
 */
export async function listDiaries(characterId: string, limit: number = 10): Promise<DiaryEntry[]> {
  const response = await fetch(`${API_BASE}/api/v1/diary/list?character_id=${characterId}&limit=${limit}`);

  if (!response.ok) {
    throw new Error(`Failed to list diaries: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 获取最新日记
 * @param characterId - 角色 ID
 */
export async function getLatestDiary(characterId: string): Promise<DiaryEntry | null> {
  const response = await fetch(`${API_BASE}/api/v1/diary/latest?character_id=${characterId}`);

  if (!response.ok) {
    throw new Error(`Failed to get latest diary: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 根据路径获取日记详情
 * @param path - 文件相对路径，例如 "{uuid}/daily/2025-01-23-14_30_52.txt"
 */
export async function getDiaryByPath(path: string): Promise<DiaryEntry> {
  const response = await fetch(`${API_BASE}/api/v1/diary/${path}`);

  if (!response.ok) {
    throw new Error(`Failed to get diary: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 创建日记
 */
export async function createDiary(data: {
  character_id: string;
  date: string;
  content: string;
  tag?: string;
}): Promise<{ message: string; diary: DiaryEntry }> {
  const response = await fetch(`${API_BASE}/api/v1/diary/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error(`Failed to create diary: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 更新日记内容
 * @param path - 文件相对路径
 * @param content - 新的日记内容（包含 Tag 行）
 */
export async function updateDiary(
  path: string,
  content: string
): Promise<{ message: string; diary: DiaryEntry }> {
  // 根据路径读取当前内容，然后使用 AI 更新
  const currentDiary = await getDiaryByPath(path);

  // 使用 AI 更新接口
  const response = await fetch(`${API_BASE}/api/v1/diary/ai-update`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      target: currentDiary.content,
      replace: content,
      character_id: currentDiary.character_id
    }),
  });

  if (!response.ok) {
    throw new Error(`Failed to update diary: ${response.statusText}`);
  }

  const result = await response.json();

  // 返回更新后的日记
  return {
    message: result.message,
    diary: await getDiaryByPath(result.path)
  };
}

/**
 * 删除日记
 * @param path - 文件相对路径
 */
export async function deleteDiary(path: string): Promise<{ message: string; path: string }> {
  const response = await fetch(`${API_BASE}/api/v1/diary/${path}`, {
    method: 'DELETE',
  });

  if (!response.ok) {
    throw new Error(`Failed to delete diary: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 获取所有日记本名称列表
 */
export async function listDiaryNames(): Promise<{ names: string[] }> {
  const response = await fetch(`${API_BASE}/api/v1/diary/names`);

  if (!response.ok) {
    throw new Error(`Failed to list diary names: ${response.statusText}`);
  }

  return response.json();
}

/**
 * 从文件路径提取日期
 * @param path - 文件路径，例如 "{uuid}/daily/2025-01-23-14_30_52.txt"
 */
export function extractDateFromPath(path: string): Date {
  const filename = path.split('/').pop() || '';
  // Remove .txt extension
  const nameWithoutExt = filename.replace('.txt', '');

  // Format: "2025-01-23-14_30_37" -> extract "2025-01-23"
  let datePart = nameWithoutExt.split('_')[0]; // "2025-01-23-14"

  // If date part contains extra time segment, remove it
  if (datePart.split('-').length > 3) {
    const parts = datePart.split('-');
    datePart = `${parts[0]}-${parts[1]}-${parts[2]}`; // Keep only YYYY-MM-DD
  }

  const date = new Date(datePart);
  if (isNaN(date.getTime())) {
    // If still invalid, return current date
    return new Date();
  }
  return date;
}

/**
 * Group diaries by month for timeline display
 * @param diaries - Array of diary entries to group
 * @returns Array of diary groups sorted by date (newest first)
 */
export function groupDiariesByMonth(diaries: DiaryEntry[]): DiaryGroup[] {
  const groupMap: DiaryGroupMap = {};

  diaries.forEach((diary) => {
    const date = extractDateFromPath(diary.path);
    const year = date.getFullYear();
    const month = date.getMonth() + 1; // 1-12
    const key = `${year}-${month}`;

    if (!groupMap[key]) {
      groupMap[key] = [];
    }
    groupMap[key].push(diary);
  });

  // Convert to array and sort by date (newest first)
  const groups: DiaryGroup[] = Object.entries(groupMap)
    .map(([key, diaries]) => {
      const [year, month] = key.split('-').map(Number);
      return {
        year,
        month,
        monthLabel: `${year}年${month}月`,
        diaries,
        count: diaries.length,
        expanded: false, // Default collapsed
      };
    })
    .sort((a, b) => {
      // Sort descending: newest month first
      if (a.year !== b.year) return b.year - a.year;
      return b.month - a.month;
    });

  // Expand the first (newest) group by default
  if (groups.length > 0) {
    groups[0].expanded = true;
  }

  return groups;
}
