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

# 디자인 시스템 가이드라인

## 핵심 원칙
- **바이브 디자인:** 전문적이고 고성능인 AI 도구의 "느낌"에 집중한다.
- **글래스모피즘:** 반투명 표면과 블러로 깊이감을 만든다.
- **다이내믹 에너지:** 그라데이션과 은은한 글로우로 STT 엔진의 "지능"을 암시한다.
- **명료성:** 화려한 비주얼 속에서도 핵심 동작(업로드/전사)은 분명하게 드러나야 한다.

## 컴포넌트 규칙

### 컨테이너
- **메인 래퍼:** 최대 너비 1000px, 가운데 정렬.
- **글래스 카드:** `surface` 색상에 `glass` 효과와 `border_radius` 적용.

### 버튼
- **Primary:** `gradient` 배경 + 흰색 텍스트 + 강한 호버 글로우.
- **Secondary:** 투명 배경 + 얇은 테두리 + 은은한 호버 배경.

### 입력 요소
- **파일 드롭존:** 큰 점선 테두리 영역, 드래그오버 시 `primary` 색으로 변함.
- **텍스트 입력:** 어두운 배경, 포커스 시 `primary` 테두리.

### 트랜지션
- 모든 인터랙티브 요소는 색상·변형에 `0.3s ease` 트랜지션을 적용한다.
- 페이지 로드 시 은은한 페이드인 + 슬라이드업 애니메이션.

## AMD 하드웨어 팁 (UI 기능)
- CUDA가 표준이지만, 이 도구는 Ryzen 프로세서에서 특히 뛰어난 성능을 내는 CPU `int8`에 최적화되어 있음을 알리는 섹션/툴팁을 포함한다.
