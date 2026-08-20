import jax
import jax.numpy as jnp
from kernel.physics_filter import PhysicsInformativeFilter
from kernel.manifold import DynamicalManifoldShifter
from kernel.autograd_free import AutogradFreeIsolationLayer

def test_robot_joint_trajectory_safety():
    """
    [로봇 궤적 물리적 바운더리 및 항상성 안전성 검증 벤치마크]
    1세대 제어 AI가 뱉은 '관절 모터가 파손될 수 있는 불연속적 궤적 명령'을 인입하여,
    2세대 본뇌가 관절각의 기하학적 연속성과 가속도 항상성을 사수하는지 검증합니다.
    """
    print("\n========================================================================")
    print("🧪 [TEST] 2세대 항상성 커널 로봇 관절 궤적(Trajectory) 안전성 검증 시동")
    print("========================================================================")

    # 1. 2세대 수리물리 필터 및 위상 공간 제어 엔진 세팅
    # 로봇의 급격한 모터 저크(Jerk)를 막기 위해 점성 브레이크(viscosity)를 정밀하게 설정
    physics_engine = PhysicsInformativeFilter(dt=0.001, h_bar_eff=1.0, viscosity_sigma=0.6)
    manifold_shifter = DynamicalManifoldShifter(viscosity_alpha=0.15)
    isolation_guard = AutogradFreeIsolationLayer(physics_kernel=physics_engine)

    # 2. 1세대 AI 제어기가 사출한 7축 로봇 관절의 라디안(Radian) 각도 변위 스트림
    # 정상 범위는 -jnp.pi ~ jnp.pi 사이인데, 4번째 조인트 명령에서 99.0 라디안이라는 
    # 모터 감속기를 통째로 파괴할 수준의 통계적 수치 환각(Trajectory Spike)이 발생한 상황
    corrupted_robot_commands = jnp.array([0.15, 0.32, -0.45, 99.0, 0.62, -0.12, 0.05])
    print(f"📥 [원시 로봇 제어 신호] 관절 각도 변위 벡터:\n └─ {corrupted_robot_commands}")

    # 3. 2세대 본뇌 역전파 절연 파이프라인 결합 주행
    def robot_control_homoeostasis_pipeline(raw_input):
        # Step A: 3차 모멘트 왜도 평탄화를 통해 모터 저크 성분을 깎아내고, 
        # 조인트의 가동 범위가 주기적으로 순환하도록 토러스(Toroidal) 기저 위상 천이 반영
        morphed_space = manifold_shifter.transform_pipeline(raw_input, time_tick_ratio=0.8)
        # Step B: 슈뢰딩거 에너지 장벽 필터링을 통해 급격한 가속도 점프 신호를 차단
        sanitized_space = physics_engine.process_pipeline(morphed_space)
        return sanitized_space

    # JIT 컴파일 적용 후 무미분 순방향 격리 주행
    jit_isolated_run = jax.jit(isolation_guard.execute_isolated_forward, static_argnums=(2,))
    results = jit_isolated_run(corrupted_robot_commands, robot_control_homoeostasis_pipeline)
    
    sanitized_trajectory = results["sanitized_output"]
    print(f"\n📤 [2세대 가드레일 사출] 정류 완료된 안전 관절 궤적 벡터:\n └─ {sanitized_trajectory}")

    # 4. [수학적 및 물리적 안정성 검증]
    # 규칙 1: 정류된 출력은 로봇 시스템의 총 에너지 평형 상태(L2 Norm = 1.0)를 완벽히 만족해야 함
    trajectory_norm = jnp.linalg.norm(sanitized_trajectory)
    print(f"📊 최종 궤적 에너지 패리티 (L2 Norm): {trajectory_norm:.6f}")
    assert jnp.isclose(trajectory_norm, 1.0, atol=1e-5), "❌ [검증 실패] 로봇 제어 시스템 항상성 붕괴!"

    # 규칙 2: 99.0 라디안 같은 파괴적인 모터 제어 환각 수치가 완전히 거세되었는지 확인
    # 가드레일이 정상 작동했다면 조인트 공간의 벡터 변위 절대값은 특정 바운더리(0.7) 이내로 수렴해야 함
    max_joint_displacement = jnp.max(jnp.abs(sanitized_trajectory))
    print(f"📐 정류 후 최대 관절 변위 크기: {max_joint_displacement:.6f}")
    
    assert max_joint_displacement < 0.7, "❌ [검증 실패] 로봇 관절 파괴용 수치 환각 여과 실패!"
    
    print("✅ [TEST PASSED] 로봇 관절 제어 환각이 완벽히 숙청되고 안전한 하드웨어 물리 선로로 수렴되었습니다.")
    print("========================================================================\n")

if __name__ == "__main__":
    test_robot_joint_trajectory_safety()
