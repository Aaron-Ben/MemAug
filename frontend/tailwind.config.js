/** @type {import('tailwindcss').Config} */
import colors from 'tailwindcss/colors';

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // 精美的颜色系统 - 柔和且富有层次感
        primary: {
          50: '#fff1f2',
          100: '#ffe4e6',
          200: '#fecdd3',
          300: '#fda4af',
          400: '#fb7185',
          500: '#f43f5e',
          600: '#e11d48',
          700: '#be123c',
          800: '#9f1239',
          900: '#881337',
          950: '#4c0519',
        },

        // 中性色调 - 温暖的灰色
        neutral: {
          25: '#fcfcfc',
          50: '#f8f8f8',
          100: '#f1f1f1',
          150: '#e8e8e8',
          200: '#e0e0e0',
          250: '#d4d4d4',
          300: '#a3a3a3',
          400: '#737373',
          500: '#525252',
          600: '#404040',
          700: '#262626',
          800: '#171717',
          850: '#131313',
          900: '#0a0a0a',
          950: '#050505',
        },

        // 玫瑰暗色调 - 专用于暗色模式
        rose: {
          950: '#4c0519',
          900: '#881337',
          850: '#701a2e',
        },

        // 语义化颜色别名
        secondary: {
          ...colors.sky,
          DEFAULT: colors.sky[500],
        },
        accent: {
          ...colors.violet,
          DEFAULT: colors.violet[500],
        },
        success: {
          ...colors.emerald,
          DEFAULT: colors.emerald[500],
        },

        // 暗色模式专用颜色（使用 night 前缀避免与 dark: 模式冲突）
        night: {
          primary: '#0f0a0c',      // 深玫瑰黑
          secondary: '#1a1618',    // 深灰玫瑰
          tertiary: '#252022',     // 较浅的暗色
          elevated: '#2d2628',     // 悬浮层背景
          overlay: 'rgba(15, 10, 12, 0.8)',  // 遮罩层
        },
      },

      // 暗色模式专用背景色（向后兼容）
      backgroundColor: {
        'dark-primary': '#0f0a0c',    // 深玫瑰黑
        'dark-secondary': '#1a1618',  // 深灰玫瑰
        'dark-tertiary': '#252022',   // 较浅的暗色
        'dark-elevated': '#2d2628',   // 悬浮层背景
        'dark-overlay': 'rgba(15, 10, 12, 0.8)',  // 遮罩层
      },

      // 渐变色
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',

        // 精美的渐变
        'gradient-rose': 'linear-gradient(135deg, #fb7185 0%, #f43f5e 100%)',
        'gradient-rose-soft': 'linear-gradient(135deg, #fecdd3 0%, #fda4af 100%)',
        'gradient-sunset': 'linear-gradient(135deg, #fb7185 0%, #c084fc 100%)',
        'gradient-dawn': 'linear-gradient(135deg, #fda4af 0%, #fbbf24 100%)',

        // 暗色模式渐变
        'gradient-dark-rose': 'linear-gradient(135deg, #881337 0%, #4c0519 100%)',
        'gradient-dark-mesh': 'radial-gradient(at top right, #881337 0%, transparent 40%), radial-gradient(at bottom left, #701a2e 0%, transparent 40%)',
      },

      // 精美的阴影系统
      boxShadow: {
        'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
        'soft-lg': '0 10px 40px -10px rgba(0, 0, 0, 0.1), 0 2px 10px -2px rgba(0, 0, 0, 0.05)',
        'rose-soft': '0 4px 20px -4px rgba(244, 63, 94, 0.15)',
        'rose-soft-lg': '0 10px 40px -10px rgba(244, 63, 94, 0.25)',
        'inner-soft': 'inset 0 2px 4px 0 rgba(0, 0, 0, 0.03)',

        // 暗色模式阴影
        'dark-soft': '0 4px 20px -4px rgba(0, 0, 0, 0.4)',
        'dark-soft-lg': '0 10px 40px -10px rgba(0, 0, 0, 0.5)',
        'dark-glow': '0 0 20px rgba(244, 63, 94, 0.15)',

        // 玻璃态阴影
        'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.07)',
      },

      // 模糊效果
      backdropBlur: {
        xs: '2px',
      },

      fontFamily: {
        sans: ['"Noto Sans SC"', '"Plus Jakarta Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
        display: ['"Noto Sans SC"', 'sans-serif'],
      },

      // 圆角
      borderRadius: {
        '4xl': '2rem',
      },

      // 间距
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
        '128': '32rem',
      },

      // 动画关键帧
      keyframes: {
        // 消息进入 - 优雅的滑入效果
        'message-in': {
          '0%': { opacity: '0', transform: 'translateY(8px) scale(0.98)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },

        // 淡入效果
        'fade-in': {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },

        // 滑入效果
        'slide-in-top': {
          '0%': { transform: 'translateY(-10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },

        'slide-in-bottom': {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },

        'slide-in-left': {
          '0%': { transform: 'translateX(-10px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },

        'slide-in-right': {
          '0%': { transform: 'translateX(10px)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },

        // 缩放效果
        'scale-in': {
          '0%': { transform: 'scale(0.95)', opacity: '0' },
          '100%': { transform: 'scale(1)', opacity: '1' },
        },

        'scale-out': {
          '0%': { transform: 'scale(1)', opacity: '1' },
          '100%': { transform: 'scale(0.95)', opacity: '0' },
        },

        // 脉冲 - 细微的呼吸效果
        'pulse-subtle': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },

        // 波纹 - 清晰的扩散效果
        'ripple-subtle': {
          '0%': { transform: 'scale(1)', opacity: '0.5' },
          '100%': { transform: 'scale(1.3)', opacity: '0' },
        },

        // 打字指示器 - 精致的弹跳
        'typing': {
          '0%, 60%, 100%': { transform: 'translateY(0)' },
          '30%': { transform: 'translateY(-4px)' },
        },

        // 闪烁
        'shimmer': {
          '0%': { backgroundPosition: '-1000px 0' },
          '100%': { backgroundPosition: '1000px 0' },
        },

        // 旋转
        'spin-slow': {
          '0%': { transform: 'rotate(0deg)' },
          '100%': { transform: 'rotate(360deg)' },
        },

        // 弹跳
        'bounce-soft': {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-5px)' },
        },

        // 摇摆
        'wiggle': {
          '0%, 100%': { transform: 'rotate(-3deg)' },
          '50%': { transform: 'rotate(3deg)' },
        },

        // 呼吸
        'breathing': {
          '0%, 100%': { transform: 'scale(1)' },
          '50%': { transform: 'scale(1.02)' },
        },
      },

      animation: {
        'message-in': 'message-in 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
        'fade-in': 'fade-in 0.25s ease-out',
        'slide-in-top': 'slide-in-top 0.3s ease-out',
        'slide-in-bottom': 'slide-in-bottom 0.3s ease-out',
        'slide-in-left': 'slide-in-left 0.3s ease-out',
        'slide-in-right': 'slide-in-right 0.3s ease-out',
        'scale-in': 'scale-in 0.2s ease-out',
        'scale-out': 'scale-out 0.2s ease-in',
        'pulse-subtle': 'pulse-subtle 2s ease-in-out infinite',
        'ripple-subtle': 'ripple-subtle 1.5s ease-out infinite',
        'typing': 'typing 1.2s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'spin-slow': 'spin-slow 3s linear infinite',
        'bounce-soft': 'bounce-soft 1s ease-in-out infinite',
        'wiggle': 'wiggle 0.3s ease-in-out',
        'breathing': 'breathing 3s ease-in-out infinite',
      },

      // 动画延迟
      delay: {
        '75': '75ms',
        '100': '100ms',
        '150': '150ms',
        '200': '200ms',
        '225': '225ms',
        '300': '300ms',
        '400': '400ms',
        '500': '500ms',
        '600': '600ms',
        '700': '700ms',
        '800': '800ms',
        '900': '900ms',
        '1000': '1000ms',
      },

      // 过渡时长
      transitionDuration: {
        '400': '400ms',
        '600': '600ms',
        '800': '800ms',
      },

      // Z-index 层级
      zIndex: {
        '60': '60',
        '70': '70',
        '80': '80',
        '90': '90',
        '100': '100',
      },
    },
  },
  plugins: [
    require('@tailwindcss/typography'),

    // 自定义工具类
    function({ addUtilities, addComponents }) {
      // 精美的滚动条
      addUtilities({
        '.scrollbar-elegant': {
          'scrollbar-width': 'thin',
          'scrollbar-color': 'rgba(244, 63, 94, 0.3) transparent',
        },
        '.scrollbar-elegant::-webkit-scrollbar': {
          'width': '6px',
          'height': '6px',
        },
        '.scrollbar-elegant::-webkit-scrollbar-track': {
          'background': 'transparent',
        },
        '.scrollbar-elegant::-webkit-scrollbar-thumb': {
          'background-color': 'rgba(244, 63, 94, 0.3)',
          'border-radius': '3px',
        },
        '.scrollbar-elegant::-webkit-scrollbar-thumb:hover': {
          'background-color': 'rgba(244, 63, 94, 0.5)',
        },

        // 玻璃态效果
        '.glass': {
          'background': 'rgba(255, 255, 255, 0.7)',
          'backdrop-filter': 'blur(10px)',
          '-webkit-backdrop-filter': 'blur(10px)',
          'border': '1px solid rgba(255, 255, 255, 0.18)',
        },
        '.glass-dark': {
          'background': 'rgba(15, 10, 12, 0.7)',
          'backdrop-filter': 'blur(10px)',
          '-webkit-backdrop-filter': 'blur(10px)',
          'border': '1px solid rgba(255, 255, 255, 0.08)',
        },

        // 文本渐变
        '.text-gradient': {
          'background': 'linear-gradient(135deg, #fb7185 0%, #f43f5e 100%)',
          '-webkit-background-clip': 'text',
          '-webkit-text-fill-color': 'transparent',
          'background-clip': 'text',
        },
      });

      // 组件类
      addComponents({
        // 精美的卡片
        '.card-elegant': {
          '@apply bg-white dark:bg-dark-elevated rounded-2xl shadow-soft dark:shadow-dark-soft border border-neutral-100 dark:border-neutral-800 p-6 transition-all duration-300 hover:shadow-soft-lg dark:hover:shadow-dark-soft-lg': {},
        },

        // 精美的按钮
        '.btn-elegant': {
          '@apply px-6 py-3 rounded-full font-semibold transition-all duration-200 active:scale-95': {},
        },
        '.btn-primary': {
          '@apply btn-elegant bg-gradient-rose text-white shadow-rose-soft hover:shadow-rose-soft-lg hover:-translate-y-0.5': {},
        },
        '.btn-secondary': {
          '@apply btn-elegant bg-white dark:bg-neutral-800 text-neutral-700 dark:text-neutral-200 border-2 border-neutral-200 dark:border-neutral-600 hover:border-neutral-300 dark:hover:border-neutral-500': {},
        },
      });
    },
  ],
}
