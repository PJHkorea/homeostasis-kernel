import jax
import jax.numpy as jnp
from kernel.physics_filter import PhysicsInformativeFilter
from kernel.manifold import DynamicalManifoldShifter
from kernel.autograd_free import AutogradFreeIsolationLayer

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
    # 0.5 근방에서 정밀하게 부품이 조립되어야 하는데, 중간에 888.0이라는 파괴적 튐(환각)이 발생한 상황
    corrupted_cad_stream = jnp.array([0.501, 0.498, 0.502, 888.0, 0.499, -555.0, 0.503])
    print(f"📥 [원시 CAD 스트림 입력] 오차 폭발 징후 벡터:\n └─ {corrupted_cad_stream}")

    # 3. 2세대 본뇌 파이프라인 결합 주행
    # 공간 왜도를 깎아내고(Manifold), 물리 가드레일을 통과(Physics)시키는 결합 함수 정의
    def total_homoeostasis_pipeline(raw_input):
        # Step A: 구면-토러스 기저 모핑 및 3차 모멘트 왜도 평탄화 처리 (t=0.5 분크 레일)
        morphed_space = manifold_shifter.transform_pipeline(raw_input, time_tick_ratio=0.5)
        # Step B: 슈뢰딩거 노치 및 카시미르 압착 물리 숙청
        sanitized_space = physics_engine.process_pipeline(morphed_space)
        return sanitized_space

       # [리팩토링] JIT 컴파일 적용 후 무미분 순방향 격리 주행 집행
    jit_isolated_run = jax.jit(isolation_guard.execute_isolated_forward, static_argnums=(2,))
    results = jit_isolated_run(corrupted_cad_stream, total_homoeostasis_pipeline)
    
    # 가속기 하드웨어 버퍼 동기화로 런타임 상수 고착화
    sanitized_output = results["sanitized_output"]
    sanitized_output.block_until_ready()
    
    print(f"\n📤 [2세대 가드레일 사출] 위상 정류 완료 다양체 벡터:\n └─ {sanitized_output}")

    # 4. [수학적 무결성 및 기하학적 수렴성 엄밀 검증]
    # 법칙 A: 최종 출력 다양체는 항상성 규칙에 의해 L2 Norm = 1.0 평형을 완벽히 사수해야 함
    # (우리가 커스텀 빌드한 L2 퓨즈드 셔플 루프의 정합성을 최종 증명합니다)
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
    test_cad_geometric_convergence()

