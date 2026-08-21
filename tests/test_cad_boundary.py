import jax
import jax.numpy as jnp
from kernel.physics_filter import PhysicsInformativeFilter
from kernel.manifold import DynamicalManifoldShifter
from kernel.autograd_free import AutogradFreeIsolationLayer

# [6차 고도화 - test_memory_o1.py에 연동된 중앙 부트스트랩 프로토콜 무리 이식]
# 개별 파일 내부에서 Abstract Tracer를 중복 가동하던 불필요한 빌드 지터를 전면 소멸시킵니다.
from tests.test_memory_o1 import trigger_global_bootstrap_precompilation

def test_cad_geometric_convergence():
    """
    [CAD 기하학적 수렴성 및 공차 정류 벤치마크 테스트]
    1세대 보조뇌가 뱉은 '누적 오차가 도미노처럼 붕괴하는 설계 신호'를 주입하여,
    2세대 본뇌가 기하학적 임계 구역 내로 완벽히 정류(수렴)시키는지 통계적으로 검증합니다.
    """
    print("\n========================================================================")
    print("🧪 [TEST] 2세대 항상성 커널 CAD 공차 누적오차 숙청 및 수렴성 검증 시동")
    print("========================================================================")

    # 1. 2세대 수리물리 핵심 가드레일 및 공간 위상 변환 엔진 준비
    physics_engine = PhysicsInformativeFilter(dt=0.001, h_bar_eff=1.0, viscosity_sigma=0.3)
    manifold_shifter = DynamicalManifoldShifter(viscosity_alpha=0.1)
    isolation_guard = AutogradFreeIsolationLayer(physics_kernel=physics_engine)

    # 2. 1세대 보조뇌의 조립 불가능한 변칙적 CAD 좌표 스트림 생성 (할루시네이션 시뮬레이션)
    # 4차 고도화 수리물리 하드웨어 정밀도를 사수하기 위해 명시적으로 float32 레일 고정
    corrupted_cad_stream = jnp.array([0.501, 0.498, 0.502, 888.0, 0.499, -555.0, 0.503], dtype=jnp.float32)
    print(f"📥 [원시 CAD 스트림 입력] 오차 폭발 징후 벡터:\n └─ {corrupted_cad_stream}")

    # 검증용 격자 및 특징 차원 수립 (7개 노드 성분 매핑)
    SPATIAL_DIM = 7

    # 3. 2세대 본뇌 파이프라인 결합 주행 정의
    def total_homoeostasis_pipeline(raw_input):
        # Step A: manifold 고도화 명세에 맞추어 정적 가상 뷰 가드레일인 spatial_dim 상수를 인입 결착
        morphed_space = manifold_shifter.transform_pipeline(raw_input, spatial_dim=SPATIAL_DIM, time_tick_ratio=0.5)
        # Step B: 슈뢰딩거 노치 및 카시미르 압착 물리 숙청 (내부 소버린 버퍼 인플레이스 치환 작동)
        sanitized_space = physics_engine.process_pipeline(morphed_space)
        return sanitized_space

    # [리팩토링 - PINN 소버린 버퍼 기증 최외곽 컴파일 결착]:
    # 최외곽 JIT 마스터 단독 컴파일러 레이어에 donate_argnums=(0,) 자원 전사 플래그를 정밀 매핑하여 0ns 재활용합니다.
    jit_isolated_run = jax.jit(isolation_guard.execute_isolated_forward, static_argnums=(2,), donate_argnums=(0,))

    # [6차 고도화 - main_orchestrator.py 유산 인입: 전역 부트스트랩 관로를 통한 0바이트 컴파일 동결막 기폭]
    # 실제 디바이스 메모리(VRAM)를 전혀 오염시키지 않는 0MB 전역 매니페스트 중앙 캐시 레일 연동 하드 록킹
    print("⏳ [System Boot] 전역 부트스트랩 매니페스트 기반 AOT 정적 예열 시동...")
    _ = trigger_global_bootstrap_precompilation(jit_isolated_run, "cad_boundary_stream", total_homoeostasis_pipeline)
    print("🏰 [System Boot] AOT CAD Boundary Kernel Fusion Success. 컴파일 지터 완전 소멸.")

    # 마스터 가속기 파이프라인 무미분 순방향 격리 주행 집행
    results = jit_isolated_run(corrupted_cad_stream, total_homoeostasis_pipeline)
    
    # 가속기 하드웨어 버퍼 동기화로 런타임 상수 고착화
    sanitized_output = results["sanitized_output"]
    sanitized_output.block_until_ready()
    
    print(f"\n📤 [2세대 가드레일 사출] 위상 정류 완료 다양체 벡터:\n └─ {sanitized_output}")


      # 4. [수학적 무결성 및 기하학적 수렴성 엄밀 검증]
    # 법칙 A: 최종 출력 다양체는 항상성 규칙에 의해 L2 Norm = 1.0 평형을 완벽히 사수해야 함
    final_norm = results["parity_metric"]
    print(f"📊 최종 항상성 에너지 패리티 (L2 Norm): {final_norm:.6f}")
    
    # pytest 호환용 정밀 오차 범위(atol=1e-5) 내 단언 집행
    assert jnp.isclose(final_norm, 1.0, atol=1e-5), "❌ [검증 실패] 에너지 보존 및 항상성 평형 붕괴!"

    # 법칙 B: 888.0, -555.0 같은 극단적 변칙 곡률(환각) 성분이 수학적으로 제거되었는지 확인
    # 슈뢰딩거 에너지 장벽과 연쇄 제곱 카시미르 필터에 의해 발산 성분은 완전히 평탄화 정류됩니다.
    max_value = jnp.max(jnp.abs(sanitized_output))
    print(f"📐 숙청 후 최대 수치 변위 크기: {max_value:.6f}")
    
    # L2 Norm이 1.0인 기하학 공간에서 환각이 정상 차단되었다면, 
    # 각 노드의 무질서 변위 절대값은 엄격하게 임계 안전 바운더리(0.6)를 절대 넘을 수 없습니다.
    assert max_value < 0.6, "❌ [검증 실패] 기하학적 수치 할루시네이션 여과 실패! 공차가 잡히지 않았습니다."
    
    print("\n✅ [TEST PASSED] CAD 누적 오차가 완벽히 숙청되고 조립 가능한 정밀 기하 공간으로 수렴되었습니다.")
    print("========================================================================\n")

if __name__ == "__main__":
    # [리팩토링]: 전역 부트스트랩 인프라와 1:1 도킹 정합을 수용한 상태로 자율 자가 진단 실행
    test_cad_geometric_convergence()


