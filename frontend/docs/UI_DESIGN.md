# UI 设计系统文档

## 概述

本项目采用了一套现代化的 UI 设计系统，支持亮色/暗色/自动三种主题模式，使用柔和的渐变、玻璃态效果和精致的动画来创造优雅的用户体验。

## 主题系统

### 主题模式

- **Light（亮色）**: 清新的白色背景，搭配温暖的玫瑰色调
- **Dark（暗色）**: 深邃的玫瑰黑背景，保持视觉舒适度
- **Auto（自动）**: 根据系统偏好自动切换

### 使用方法

```tsx
import { useTheme } from '@/contexts';

function MyComponent() {
  const { theme, setTheme, isDark } = useTheme();

  return (
    <button onClick={() => setTheme('dark')}>
      切换到暗色模式
    </button>
  );
}
```

### 主题切换按钮

```tsx
import { ThemeToggleButton, CompactThemeToggle } from '@/components/ui';

// 完整的主题切换按钮（悬浮时显示预览）
<ThemeToggleButton variant="floating" size="md" />

// 紧凑版本
<CompactThemeToggle />

// 内联版本
<ThemeToggleButton variant="inline" showLabel />
```

## 颜色系统

### 主色调

- **Primary（玫瑰红）**: `rose` - 用于主要操作、强调元素
- **Secondary（天空蓝）**: `sky` - 用于次要操作、信息提示
- **Accent（紫罗兰）**: `violet` - 用于装饰、特殊状态
- **Success（翡翠绿）**: `emerald` - 用于成功状态

### 中性色

- **Neutral（温暖灰）**: 替代默认的 slate，提供更温暖的视觉体验

### 暗色模式专用色

- `dark-primary`: #0f0a0c - 深玫瑰黑
- `dark-secondary`: #1a1618 - 深灰玫瑰
- `dark-tertiary`: #252022 - 较浅的暗色
- `dark-elevated`: #2d2628 - 悬浮层背景

## 设计效果

### 渐变效果

```tsx
// 主渐变
bg-gradient-rose         // from-rose-400 to-pink-500
bg-gradient-rose-soft    // from-fecdd3 to-fda4af
bg-gradient-sunset       // from-rose-400 to-c084fc
bg-gradient-dawn         // from-fda4af to-fbbf24

// 暗色模式渐变
bg-gradient-dark-rose    // from-881337 to-4c0519
bg-gradient-dark-mesh    // 复杂的径向渐变组合
```

### 阴影系统

```tsx
shadow-soft              // 柔和的阴影
shadow-soft-lg           // 较大的柔和阴影
shadow-rose-soft         // 玫瑰色阴影
shadow-rose-soft-lg      // 较大的玫瑰色阴影

// 暗色模式
shadow-dark-soft         // 暗色模式柔和阴影
shadow-dark-soft-lg      // 暗色模式较大阴影
shadow-dark-glow         // 暗色模式发光效果
```

### 玻璃态效果

```tsx
// 使用 Tailwind 类
glass-light              // 亮色模式玻璃态
glass-dark               // 暗色模式玻璃态

// 自定义
bg-white/80 dark:bg-dark-elevated/80 backdrop-blur-sm
```

## 动画效果

### 内置动画

- `message-in`: 消息进入动画（滑入 + 淡入）
- `fade-in`: 淡入效果
- `slide-in-*`: 四个方向的滑入效果
- `scale-in/out`: 缩放效果
- `pulse-subtle`: 细微的脉冲
- `typing`: 打字指示器动画
- `shimmer`: 闪光效果
- `breathing`: 呼吸效果

### 使用示例

```tsx
<div className="animate-fade-in">淡入效果</div>
<div className="animate-message-in">消息进入</div>
<div className="animate-bounce-soft">弹跳</div>
```

### 动画延迟

可用的延迟值：
- `delay-75` 到 `delay-1000`（75ms 到 1000ms）

## 组件样式指南

### 按钮

```tsx
// 主要按钮
<button className="btn-elegant btn-primary">
  确认
</button>

// 次要按钮
<button className="btn-elegant btn-secondary">
  取消
</button>
```

### 卡片

```tsx
<div className="card-elegant">
  精美的卡片内容
</div>
```

### 消息气泡

用户消息使用精美的渐变效果，AI 消息使用玻璃态效果：

```tsx
// 用户消息
<UserMessageBubble content="你好" />

// AI 消息
<AIMessageBubble content="你好呀！" />
```

## 响应式设计

项目使用 Tailwind 的响应式工具类：

- `sm:` 640px 及以上
- `md:` 768px 及以上
- `lg:` 1024px 及以上
- `xl:` 1280px 及以上

示例：
```tsx
<div className="text-sm md:text-base lg:text-lg">
  响应式文本大小
</div>
```

## 最佳实践

1. **主题感知**: 始终考虑亮色和暗色两种模式
   ```tsx
   className="bg-white dark:bg-neutral-800 text-neutral-800 dark:text-neutral-100"
   ```

2. **使用语义化颜色**: 优先使用 `primary`、`secondary` 等语义化颜色

3. **一致动画**: 使用内置动画而非自定义 CSS

4. **过渡效果**: 为交互元素添加过渡
   ```tsx
   className="transition-all duration-300 hover:scale-105"
   ```

5. **可访问性**: 确保足够的对比度，使用适当的焦点样式

## 浏览器兼容性

- 现代浏览器（Chrome、Firefox、Safari、Edge）
- 需要支持 backdrop-filter 的浏览器版本
- 降级方案：不支持 backdrop-filter 的浏览器将显示半透明背景

## 暗色模式背景图处理

### 方案说明

使用 **CSS Filter** 方案处理暗色模式背景图：

- `brightness(0.7)` - 降低亮度到 70%
- `grayscale(40%)` - 添加 40% 灰度
- `contrast(0.9)` - 稍微降低对比度

**优点**：性能好，实现简单，过渡平滑

### 使用方法

```tsx
import { Background } from '@/components/ui';

// 基础使用
<Background imageUrl="/background/image.png">
  <YourContent />
</Background>

// 自定义渐变
<Background
  imageUrl="/background/image.png"
  lightGradient="from-blue-50/30 via-white to-purple-50/20"
  darkGradient="from-slate-900 via-slate-800 to-slate-900"
  showTexture={false}
>
  <YourContent />
</Background>
```

### 简化版组件

```tsx
import { SimpleBackground } from '@/components/ui';

<SimpleBackground imageUrl="/background/image.png">
  <YourContent />
</SimpleBackground>
```

### CSS 内联方案

如果你不想使用组件，也可以直接使用内联样式：

```tsx
<div className="relative">
  {/* 背景图 */}
  <div
    className="absolute inset-0 bg-cover bg-center opacity-40"
    style={{ backgroundImage: 'url(...)' }}
  />

  {/* 暗色模式滤镜层 */}
  <div
    className="absolute inset-0 bg-cover bg-center opacity-0 dark:opacity-35"
    style={{
      backgroundImage: 'url(...)',
      filter: 'brightness(0.7) grayscale(40%) contrast(0.9)',
    }}
  />
</div>
```

## 更新日志

### 2026-03-07
- ✨ 创建完整的主题系统
- ✨ 添加主题切换按钮组件
- 🎨 扩展颜色系统和视觉效果
- 🎨 优化核心组件样式
- 🎨 添加渐变、阴影、动画效果
- 🌙 优化暗色模式背景图处理（filter 方案）
- ✨ 创建可复用的 Background 组件
- 🎭 角色管理界面完整暗色模式适配
  - CharacterManagementPage - 主页面
  - CreateCharacterModal - 创建角色模态框
  - CharacterSelector - 角色选择器
  - 添加玻璃态效果和精美动画
