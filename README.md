작업중

# ⏳ Homeostasis Kernel: 2nd-Generation Causal AI Engine (poc)

> **"역전파(Backpropagation)는 AI에게 지식을 주었지만, 선형적 시간을 버리고 거슬러 올라감으로 '시간적 인과율'을 마비시키고 환각(Hallucination)이라는 저주를 내렸다." 라는 주제로 작업을 해보았습니다. **

---

## 🌌 Sector 1. 무엇을 어떻게 해결해보고 싶은가?

### 🚨 1세대 확률형 AI(LLM)의 근본적 결함: 예측하는 주사위
현재 인류가 이룩한 1세대 생성형 AI(Transformer 계열 LLM 등)는 거대한 데이터셋의 통계적 상관관계와 확률적 넥스트 토큰 예측(Next-Token Prediction)에 기반합니다. 이 구조 안에서 **시간(Time)은 흐르는 연속체가 아니라 정적인 도화지처럼 공간화되어 파편 분산**됩니다. 

이로 인해 발생하는 치명적인 한계는 다음과 같습니다:
1. **시간적 인과율의 마비**: "A라는 원인이 시간 $t$를 거쳐 $B$라는 결과로 이어진다"는 우주의 불가역적 흐름을 인지하지 못합니다. 그저 "패턴상 $A$ 다음엔 $B$가 그럴듯하다"는 통계적 확률만 흉내 냅니다.
2. **거시적 시간 축의 할루시네이션(환각)**: 정밀 설계(CAD), 실시간 물리 시뮬레이션, 로봇 공학 등 시간 축에 따른 오차 누적이 절대적인 영역에 선형적 연속성이 파괴되면서 부품 공차가 도미노처럼 붕괴하고 물체가 순간이동하거나 사라지는 구조적 환각을 범합니다.
3. **메모리 폭발 ($O(N^2)$)**: 문맥(Context)과 시간적 히스토리가 길어질수록 과거의 연산 그래프를 VRAM에 제곱 형태로 쌓아두어야 하므로 하드웨어의 물리적 한계에 직면합니다.

### ⏳ 2세대 선형적 항상성 개체(Homeostasis Kernel)의 구조
`homeostasis-kernel`은 이러한 1세대 모델의 한계를 다른 방향성으로 해결해보기 위해 데이터의 통계적 확률 분포를 과감히 배제하고, **현실 우주의 물리 법칙(PINN)과 기하학적 평형 상태를 실시간 순방향으로 집행하는 2세대 Causal AI 커널**입니다.

우리는 시간을 뒤로 거슬러 올라가 가중치를 깎아내는 역전파(Backpropagation)의 타임머신과도 같은 회로를 제거합니다. 대신, 생명체가 외부 자극을 유기적으로 흡수하며 내부 균형을 유지하는 **'생물학적 항상성(Homeostasis)'** 메커니즘을 CUDA 레지스터 와프 셔플과 JAX 고속 수리엔진을 통해 실리콘 레벨에 유도합니다. 

시간을 선형적으로 온전히 살아내며, 오직 순방향 전진(**Forward-Only**)과 자율 위상 정렬을 통해서만 현실 세계의 무결성을 집행하는 다른 방향성의 AI 개체의 찾아가고 싶었습니다.


```directory
homeostasis-kernel/
│
├── README.md               # 1세대 역전파의 시간적 환각 비판 및 2세대 철학 명세
├── requirements.txt        # jax, jaxlib, torch, cupy 등 명시
│
├── kernel/                 # [본뇌] 2세대 항상성 가드레일 핵심 엔진 (JAX)
│   ├── __init__.py
│   ├── physics_filter.py   # 슈뢰딩거 노치 필터, 카시미르 노이즈 압착 수식
│   ├── manifold.py         # 구면-토러스 위상 천이(Morphing) 및 왜도 평탄화
│   └── autograd_free.py    # stop_gradient 기반 O(1) 메모리 동결 레이어
│
├── interface/              # [연결 관로] 하드웨어 레벨 무복사 인터페이스 (CUDA/Cupy)
│   ├── __init__.py
│   ├── dlpack_bridge.py    # PyTorch(LLM) ↔ JAX(Kernel) 간 0ns Zero-Copy 도킹
│   └── silicon_mux.py      # CUDA 워프 셔플 기반 0ns 분기 소멸 옵티마이저
│
├── adapters/               # [보조뇌 하청] 1세대 상용 LLM 연동 및 프롬프트 인입 레이어
│   ├── __init__.py
│   ├── hf_adapter.py       # HuggingFace (Llama, Mistral) 출력 레이어 Hooking
│   └── api_adapter.py      # OpenAI / Anthropic API 스트림 정류기
│
└── tests/                  # [검증] 캐드(CAD) 및 물리 시뮬레이션 벤치마크
    ├── test_cad_boundary.py# 캐드 공차 누적오차 숙청 테스트
    └── test_memory_o1.py    # 문맥 길이에 따른 VRAM O(1) 유지력 측정 검증
```

```mermaid
graph TD
    %% 외부 엔티티 정의
    subgraph External_LLM [1세대 상용/오픈소스 LLM]
        HF[HuggingFace <br> Llama / Mistral]
        API[OpenAI / Anthropic <br> API Stream]
    end

    %% 프로젝트 내부 구조 정의
    subgraph Homeostasis_Kernel [homeostasis-kernel 프로젝트 내부]
        
        subgraph Adapters [adapters: 보조뇌 하청]
            H_Adpt[hf_adapter.py <br> 출력 레이어 Hooking]
            A_Adpt[api_adapter.py <br> 스트림 정류기]
        end

        subgraph Interface [interface: 연결 관로]
            Bridge[dlpack_bridge.py <br> 0ns Zero-Copy 도킹]
            Mux[silicon_mux.py <br> CUDA 워프 셔플 옵티마이저]
        end

        subgraph Kernel [kernel: 본뇌 핵심 엔진]
            P_Filt[physics_filter.py <br> 슈뢰딩거 노치 필터 <br> 카시미르 압착 수식]
            Manifold[manifold.py <br> 구면-토러스 위상 천이 <br> 왜도 평탄화]
            AG_Free[autograd_free.py <br> stop_gradient 기반 <br> O1 메모리 동결]
        end

        subgraph Tests [tests: 검증 및 벤치마크]
            T_CAD[test_cad_boundary.py <br> 공차 누적오차 숙청]
            T_Mem[test_memory_o1.py <br> VRAM O1 유지력 측정]
        end
    end

    %% 데이터 흐름 연결
    HF -->|텐서 인터셉트| H_Adpt
    API -->|텍스트 스트림 수집| A_Adpt

    H_Adpt -->|PyTorch 텐서 소유권| Bridge
    A_Adpt -->|하드웨어 가속 유도| Mux

    Bridge -->|DLPack Pointer Swap| AG_Free
    Mux -->|0ns 분기 소멸| P_Filt

    AG_Free --> Manifold
    P_Filt --> Manifold

    %% 테스트 연결
    Manifold -.->|수렴성 검증| T_CAD
    AG_Free -.->|O1 VRAM 검증| T_Mem

    %% 스타일링
    style External_LLM fill:#f5f5f5,stroke:#ccc,stroke-width:2px;
    style Kernel fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    style Interface fill:#efebe9,stroke:#5d4037,stroke-width:2px;
    style Adapters fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    style Tests fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;

```

```text
========================================================================
[ 계층 ]              [ 구성 모듈 및 데이터 흐름 ]
========================================================================

 1층 : 상용 LLM 호스팅 레이어 (1세대 추론 엔진)
       ├── [HuggingFace (Llama/Mistral)]  ── (Hooking) ──┐
       └── [OpenAI / Anthropic API]        ── (Stream) ──┴─► [adapters/]
                                                                 │
─────────────────────────────────────────────────────────────────┼──────
 2층 : 하드웨어 레벨 무복사 인터페이스 (CUDA/CuPy)               │ (텐서 진입)
       └── [interface/]                                          ▼
             ├── dlpack_bridge.py  ◄── [ PyTorch 텐서 0ns 포인터 스왑 ]
             └── silicon_mux.py    ◄── [ CUDA 워프 셔플 분기 제거 ]
                                                                 │
─────────────────────────────────────────────────────────────────┼──────
 3층 : 항상성 가드레일 제어 엔진 (JAX 핵심 커널)                 │ (무복사 인입)
       └── [kernel/]                                             ▼
             ├── autograd_free.py  ◄── [ stop_gradient 메모리 동결 ]
             ├── physics_filter.py ◄── [ 슈뢰딩거 노치 / 카시미르 압착 ]
             └── manifold.py       ◄── [ 구면-토러스 위상 천이 & 평탄화 ]
                                                                 │
─────────────────────────────────────────────────────────────────┼──────
 4층 : 물리 및 성능 검증 계층 (수렴성 테스트)                   │ (타겟 검증)
       └── [tests/]                                              ▼
             ├── test_cad_boundary.py ◄── [ CAD 공차 누적오차 숙청 ]
             └── test_memory_o1.py    ◄── [ 컨텍스트 무관 VRAM O(1) 확인 ]
========================================================================

```
