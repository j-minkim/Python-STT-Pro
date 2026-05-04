---
version: "1.0.0"
name: "Python STT Pro - Premium Vibe"

colors:
  primary: "#00F2FE" # Cyan accent
  secondary: "#4FACFE" # Blue accent
  background: "#0F172A" # Deep slate dark mode
  surface: "rgba(30, 41, 59, 0.7)" # Glassmorphism base
  text: "#F8FAFC"
  text_muted: "#94A3B8"
  accent: "#F472B6" # Pink for highlights
  success: "#10B981"
  warning: "#F59E0B"
  error: "#EF4444"

typography:
  fontFamily: "'Outfit', 'Inter', sans-serif"
  h1:
    fontSize: "3.5rem"
    fontWeight: "800"
    letterSpacing: "-0.025em"
  body:
    fontSize: "1rem"
    lineHeight: "1.6"

spacing:
  base: "16px"
  card_padding: "24px"
  border_radius: "16px"

effects:
  glass: "backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.1);"
  shadow: "0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.2);"
  gradient: "linear-gradient(135deg, #4FACFE 0%, #00F2FE 100%)"
---

# Design System Guidelines

## Core Principles
- **Vibe Design:** Focus on the "feeling" of professional, high-performance AI tools.
- **Glassmorphism:** Use translucent surfaces with blurs to create depth.
- **Dynamic Energy:** Use gradients and subtle glows to suggest the "intelligence" of the STT engine.
- **Clarity:** Despite the rich aesthetics, the main action (uploading/transcribing) must be unmistakable.

## Component Rules

### Containers
- **Main Wrapper:** Max-width 1000px, centered.
- **Glass Cards:** Use `surface` color with `glass` effect and `border_radius`.

### Buttons
- **Primary:** Use `gradient` with white text and a strong hover glow.
- **Secondary:** Transparent with thin border, subtle hover background.

### Inputs
- **File Dropzone:** Large dashed border area, turns `primary` color on drag-over.
- **Text Inputs:** Dark background, `primary` border on focus.

### Transitions
- All interactive elements should have a `0.3s ease` transition for colors and transforms.
- Page load should have a subtle fade-in and slide-up animation.

## AMD Hardware Tips (UI Feature)
- Include a specific section or tooltip highlighting that while CUDA is standard, the tool is optimized for CPU `int8` which performs exceptionally well on Ryzen processors.
