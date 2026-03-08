"""Time expression parser for natural language time references."""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pytz
from dataclasses import dataclass

from app.config.time_expressions import (
    get_time_expressions,
    parse_chinese_number,
    TimeExpression as TimeExpressionConfig,
    CHINESE_WEEKDAYS,
    ENGLISH_WEEKDAYS,
)

logger = logging.getLogger(__name__)


@dataclass
class TimeRange:
    """Time range with start and end datetime."""
    start: datetime
    end: datetime

    def __eq__(self, other):
        if not isinstance(other, TimeRange):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self):
        return hash((self.start.timestamp(), self.end.timestamp()))


@dataclass
class TimeExpression:
    """Parsed time expression result."""
    text: str
    range: TimeRange


class TimeExpressionParser:
    """
    Parser for natural language time expressions.

    Supports Chinese and English expressions with timezone awareness.
    """

    def __init__(self, locale: str = 'zh-CN', default_timezone: str = 'Asia/Shanghai'):
        """
        Initialize the time expression parser.

        Args:
            locale: Locale string (e.g., 'zh-CN', 'en-US')
            default_timezone: Default timezone for parsing (default: 'Asia/Shanghai')
        """
        self.default_timezone = pytz.timezone(default_timezone)
        self.set_locale(locale)

    def set_locale(self, locale: str):
        """Set the locale for time expressions."""
        self.locale = locale
        self.expressions = get_time_expressions(locale)
        logger.info(f"[TimeParser] Locale set to: {locale}")

    def _get_now(self) -> datetime:
        """Get current time in configured timezone."""
        return datetime.now(self.default_timezone)

    def _get_day_boundaries(self, date: datetime) -> TimeRange:
        """
        Get the start and end of a day.

        Args:
            date: Date to get boundaries for (will be converted to configured timezone)

        Returns:
            TimeRange with start and end of day
        """
        # Convert to configured timezone if not already
        if date.tzinfo is None:
            date = self.default_timezone.localize(date)
        else:
            date = date.astimezone(self.default_timezone)

        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = date.replace(hour=23, minute=59, second=59, microsecond=999999)
        return TimeRange(start=start, end=end)

    def parse(self, text: str) -> List[TimeRange]:
        """
        Parse text for all time expressions.

        Args:
            text: Text to parse for time expressions

        Returns:
            List of unique TimeRange objects found in the text
        """
        text_preview = text[:100] + ('...' if len(text) > 100 else '')
        logger.info(f"[TimeParser] ========================================")
        logger.info(f"[TimeParser] 🕐 时间解析器被调用!")
        logger.info(f"[TimeParser] 解析文本: \"{text_preview}\"")

        now = self._get_now()
        remaining_text = text
        results: List[TimeRange] = []

        # 1. Check hardcoded expressions (sorted by length, longest first)
        sorted_keys = sorted(
            self.expressions['hardcoded'].keys(),
            key=len,
            reverse=True
        )

        for expr in sorted_keys:
            if expr in remaining_text:
                config: TimeExpressionConfig = self.expressions['hardcoded'][expr]
                logger.info(f"[TimeParser] Matched hardcoded expression: \"{expr}\"")

                result = None
                if config.days is not None:
                    # 支持未来日期：负数表示未来，正数表示过去
                    if config.days < 0:
                        target_date = now + timedelta(days=-config.days)
                    else:
                        target_date = now - timedelta(days=config.days)
                    result = self._get_day_boundaries(target_date)
                elif config.type:
                    result = self._get_special_range(now, config.type)

                if result:
                    results.append(result)
                    remaining_text = remaining_text.replace(expr, '', 1)

        # 2. Check dynamic patterns
        for pattern in self.expressions['patterns']:
            regex = pattern.regex
            for match in regex.finditer(remaining_text):
                matched_text = match.group(0)
                logger.info(f"[TimeParser] Matched pattern: \"{regex.pattern}\" with text \"{matched_text}\"")

                result = self._handle_dynamic_pattern(match, pattern.type, now)
                if result:
                    results.append(result)
                    remaining_text = remaining_text.replace(matched_text, '', 1)

        # 3. Deduplicate results
        if results:
            unique_ranges = self._deduplicate_ranges(results)
            if len(unique_ranges) < len(results):
                logger.info(f"[TimeParser] Deduplicated time ranges: {len(results)} → {len(unique_ranges)}")

            logger.info(f"[TimeParser] 找到 {len(unique_ranges)} 个唯一时间表达式")
            for i, r in enumerate(unique_ranges, 1):
                logger.info(f"  [{i}] 时间范围: {r.start.strftime('%Y-%m-%d %H:%M:%S')} 到 {r.end.strftime('%Y-%m-%d %H:%M:%S')}")

            logger.info(f"[TimeParser] 时间解析完成，返回 {len(unique_ranges)} 个时间范围")
            logger.info(f"[TimeParser] ========================================")

            return unique_ranges
        else:
            logger.info("[TimeParser] 未在文本中找到时间表达式")
            logger.info(f"[TimeParser] ========================================")
            return []

    def _deduplicate_ranges(self, ranges: List[TimeRange]) -> List[TimeRange]:
        """
        Remove duplicate time ranges.

        Args:
            ranges: List of TimeRange objects

        Returns:
            Deduplicated list of TimeRange objects
        """
        unique_ranges: Dict[str, TimeRange] = {}
        for r in ranges:
            key = f"{r.start.timestamp()}|{r.end.timestamp()}"
            if key not in unique_ranges:
                unique_ranges[key] = r
        return list(unique_ranges.values())

    def _get_special_range(self, now: datetime, type_: str) -> Optional[TimeRange]:
        """
        Get special time range like 'thisWeek', 'lastMonth', etc.

        Args:
            now: Current datetime in configured timezone
            type_: Type of special range

        Returns:
            TimeRange or None if type is not recognized
        """
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now.replace(hour=23, minute=59, second=59, microsecond=999999)

        if type_ == 'thisWeek':
            # Monday as start of week
            start = start - timedelta(days=start.weekday())
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        elif type_ == 'lastWeek':
            start = start - timedelta(days=start.weekday() + 7)
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
        elif type_ == 'thisMonth':
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            # Get last day of month
            next_month = start.replace(day=28) + timedelta(days=4)  # Guaranteed to be in next month
            last_day = next_month - timedelta(days=next_month.day)
            end = last_day.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif type_ == 'lastMonth':
            # First day of last month
            if start.month == 1:
                start = start.replace(year=start.year - 1, month=12, day=1)
            else:
                start = start.replace(month=start.month - 1, day=1)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            # Last day of last month
            next_month = start.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            end = last_day.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif type_ == 'thisMonthStart':
            # 本月初 (1-10号)
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(day=10, hour=23, minute=59, second=59, microsecond=999999)
        elif type_ == 'lastMonthStart':
            # 上月初 (1-10号)
            if start.month == 1:
                start = start.replace(year=start.year - 1, month=12, day=1)
            else:
                start = start.replace(month=start.month - 1, day=1)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(day=10, hour=23, minute=59, second=59, microsecond=999999)
        elif type_ == 'lastMonthMid':
            # 上月中 (11-20号)
            if start.month == 1:
                start = start.replace(year=start.year - 1, month=12, day=11)
            else:
                start = start.replace(month=start.month - 1, day=11)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(day=20, hour=23, minute=59, second=59, microsecond=999999)
        elif type_ == 'lastMonthEnd':
            # 上月末 (21号到月底)
            if start.month == 1:
                start = start.replace(year=start.year - 1, month=12, day=21)
            else:
                start = start.replace(month=start.month - 1, day=21)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            # Get last day of that month
            next_month = start.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            end = last_day.replace(hour=23, minute=59, second=59, microsecond=999999)
        else:
            logger.warning(f"[TimeParser] Unknown special range type: {type_}")
            return None

        return TimeRange(start=start, end=end)

    def _handle_dynamic_pattern(
        self,
        match,
        type_: str,
        now: datetime
    ) -> Optional[TimeRange]:
        """
        Handle dynamic pattern matches.

        Args:
            match: Regex match object
            type_: Type of pattern (daysAgo, weeksAgo, etc.)
            now: Current datetime in configured timezone

        Returns:
            TimeRange or None if pattern cannot be handled
        """
        num_str = match.group(1) if match.lastindex and match.lastindex >= 1 else ''
        num = parse_chinese_number(num_str)

        if type_ == 'daysAgo':
            target_date = now - timedelta(days=num)
            return self._get_day_boundaries(target_date)

        elif type_ == 'weeksAgo':
            # Start of the week N weeks ago
            weeks_ago = now - timedelta(weeks=num)
            start = weeks_ago - timedelta(days=weeks_ago.weekday())
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=6, hours=23, minutes=59, seconds=59)
            return TimeRange(start=start, end=end)

        elif type_ == 'monthsAgo':
            # Start of the month N months ago
            if now.month - num <= 0:
                # Cross year boundary
                years_diff = (num - now.month) // 12 + 1
                month = 12 - (num - now.month) % 12
                start = now.replace(year=now.year - years_diff, month=month, day=1)
            else:
                start = now.replace(month=now.month - num, day=1)
            start = start.replace(hour=0, minute=0, second=0, microsecond=0)
            # Get last day of that month
            next_month = start.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            end = last_day.replace(hour=23, minute=59, second=59, microsecond=999999)
            return TimeRange(start=start, end=end)

        elif type_ == 'lastWeekday':
            weekday_char = match.group(1) if match.lastindex and match.lastindex >= 1 else ''
            weekday_map = CHINESE_WEEKDAYS if self.locale == 'zh-CN' else ENGLISH_WEEKDAYS
            target_weekday = weekday_map.get(weekday_char)

            if target_weekday is None:
                logger.warning(f"[TimeParser] Invalid weekday: {weekday_char}")
                return None

            # Find the last occurrence of this weekday
            current_weekday = now.weekday()
            days_ago = (current_weekday - target_weekday) % 7
            if days_ago == 0:
                days_ago = 7  # If today, get last week's
            last_week_date = now - timedelta(days=days_ago)
            return self._get_day_boundaries(last_week_date)

        logger.warning(f"[TimeParser] Unknown pattern type: {type_}")
        return None
