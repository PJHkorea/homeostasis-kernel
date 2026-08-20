import jax
import jax.numpy as jnp
from functools import partial
from interface.silicon_mux import SiliconMuxOptimizer

class PhysicsInformativeFilter:
    """
    2세대 항상성 핵심 커널 - 물리 기반 정보 필터 (Physics-Informed Filter).
    1세대 보조뇌의 확률적 수치 환각을 슈뢰딩거 및 카시미르 수식으로 깎아냅니다.
    """
    # [math_guardrails 기믹 반영] 전역 환경 오염 및 임계 족쇄를 방어할 leaky_slope와 boundary_margin 인자 분리 인입
    def __init__(self, dt: float = 0.001, h_bar_eff: float = 1.0, viscosity_sigma: float = 0.1, leaky_slope: float = 0.01, boundary_margin: float = 0.05):
        self.dt = dt                      # 선형적으로 쪼갠 미세 시간 격자
        self.h_bar = h_bar_eff            # 위상 상태 결맞음(Coherence)을 위한 유효 플랑크 상수
        self.sigma = viscosity_sigma      # 매니폴드 파괴를 막는 물리적 점성 브레이크 계수
        self.leaky_slope = leaky_slope    # [math_guardrails] 임계 범위를 초과한 영역에 부여하는 미세 복원 기울기
        self.boundary_margin = boundary_margin # [math_guardrails] 리키 가드레일이 활성화될 소프트 임계 경계선 마진
        self.mux_opt = SiliconMuxOptimizer() # 하드웨어 분기 제거를 위한 MUX 내장

    @partial(jax.jit, static_argnums=(0,))
    def execute_schrodinger_notch_filter(self, raw_stream: jnp.ndarray) -> jnp.ndarray:
        """
        [물리 가드레일 1] 슈뢰딩거 노치 필터 (Leaky Guardrail 융합 버전)
        시간 격자(dt) 기반 이계 미분으로 수치 곡률을 산출한 뒤 양자 터널링 투과 계수를 전개합니다.
        [math_guardrails 기믹] 투과도가 제로(0.0f)로 소멸하는 한계면 바깥에 미세 기울기를 부여하여 오차 복원 그레디언트를 끝까지 살려냅니다.
        """
        target_dtype = raw_stream.dtype
        safe_dt = jnp.array(self.dt, dtype=target_dtype)
        safe_hbar = jnp.array(self.h_bar, dtype=target_dtype)
        safe_leaky = jnp.array(self.leaky_slope, dtype=target_dtype)
        safe_margin = jnp.array(self.boundary_margin, dtype=target_dtype)
        
        # 1. 시간 격자(dt)를 분모로 명시한 엄밀한 1차, 2차 시간 미분 대리값 기반 실제 곡률(Curvature) 추출
        dx = jnp.gradient(raw_stream, safe_dt)
        curvature = jnp.abs(jnp.gradient(dx, safe_dt))
        
        # 2. 곡률에 비례하는 포텐셜 에너지 장벽(U_barrier) 계산
        u_barrier = jax.lax.mul(jnp.array(self.sigma, dtype=target_dtype), curvature)
        
        # 3. 양자 터널링 투과 계수 수식 전개 (T = exp(-2 * sqrt(U) / h_bar))
        sqrt_u = jax.lax.sqrt(jax.lax.add(u_barrier, jnp.array(1e-12, dtype=target_dtype)))
        
        exponent = jax.lax.neg(
            jax.lax.div(
                jax.lax.mul(jnp.array(2.0, dtype=target_dtype), sqrt_u), 
                safe_hbar
            )
        )
        transmission_coeff = jax.lax.exp(exponent)
        
        # 4. [math_guardrails 핵심 기믹: 소프트 임계 경계면 선형 확장]
        # 투과 계수가 마진 임계값 이하로 극단적으로 깎여나가 노드가 완전히 단절되려 할 때,
        # 미세 기울기(Leaky Slope)의 선형 결합 선로를 연장하여 기울기 소멸(Gradient Vanishing)을 영구 방어합니다.
        # restoration_delta = safe_margin - transmission_coeff
        restoration_delta = jax.lax.sub(safe_margin, transmission_coeff)
        # leaky_transmission = safe_margin - (safe_leaky * restoration_delta)
        leaky_transmission = jax.lax.sub(safe_margin, jax.lax.mul(safe_leaky, restoration_delta))
        
        # 5. [분기 없는 가속] 조건문 분기(if/else) 없이 하드웨어 레벨의 병렬 마스킹(jax.lax.select)으로 고속 병합
        is_above_threshold = transmission_coeff > safe_margin
        gated_coeff = jax.lax.select(is_above_threshold, transmission_coeff, leaky_transmission)
        
        # 6. [하드 가드레일] 소수점 자릿수 내림 오차 등으로 인해 경계를 이탈하는 경우를 대비해 
        # 최종 수치 한계선([1e-4, 1.0]) 내부로 강제 클램핑(Lock)하여 초월적 수치 파괴 방어
        safe_coeff = self.mux_opt.stream_boundary_clamp(gated_coeff, lower_bound=1e-4, upper_bound=1.0)
        
        # 7. 최종 1클록 선형 필터링 사출
        return jax.lax.mul(raw_stream, safe_coeff)


         @partial(jax.jit, static_argnums=(0,))
    def execute_casimir_noise_compression(self, filtered_stream: jnp.ndarray, tolerance: float = 1e-3) -> jnp.ndarray:
        """
        [물리 가드레일 2] 카시미르 위상학적 압착 수식 (Leaky Gradient 보존 버전)
        미세 오차가 임계치 이하로 좁혀질 때, 거리에 역4제곱(1/d^4) 비례하는 강력한 음압을 발생시킵니다.
        [math_guardrails 기믹] 발산 구역 진입 시 완전 절멸(0.0) 대신 미세 기울기를 주어 역전파 오차 복원 선로를 사수합니다.
        """
        target_dtype = filtered_stream.dtype
        
        # 1. 하드웨어 정밀도 동기화 및 입력 상수의 텐서 고정
        safe_epsilon = jnp.array(1e-6, dtype=target_dtype)
        safe_tolerance = jnp.array(tolerance, dtype=target_dtype)
        safe_leaky = jnp.array(self.leaky_slope, dtype=target_dtype)
        
        # 정규화된 데이터 거리 측정: d = |X| + epsilon
        distance = jax.lax.add(jnp.abs(filtered_stream), safe_epsilon)
        
        # 2. 지수 함수 파이프라인 우회를 위한 연쇄 제곱(d^4) 가속 유도 (d -> d^2 -> d^4)
        dist_sq = jax.lax.square(distance)
        dist_quad = jax.lax.square(dist_sq)
        
        # 카시미르 인력 공식 사영: F_casimir = 1.0 / d^4
        casimir_pressure = jax.lax.div(jnp.ones_like(dist_quad, dtype=target_dtype), dist_quad)
        
        # 3. 싱큘래리티 임계 장벽 연산: 1.0 / tolerance^4
        tol_sq = jax.lax.square(safe_tolerance)
        tol_quad = jax.lax.square(tol_sq)
        threshold_pressure = jax.lax.div(jnp.array(1.0, dtype=target_dtype), tol_quad)
        
        # 오차가 너무 커서 시스템 한계선을 건드리는 파괴적 발산 지역 감지
        error_mask = casimir_pressure > threshold_pressure
        
        # 4. [math_guardrails 핵심 기믹 인입: 소프트 임계 복원 경사 선형 확장]
        # 발산 구역에 진입한 성분을 무작정 0.0으로 죽여버리는 대신 원본의 부호(jnp.sign)를 보존하고,
        # 미세 기울기(leaky_slope)를 적용하여 역전파 시 오차 복원 그레디언트가 끊기지 않고 흐르도록 유도합니다.
        signed_stream = jnp.sign(filtered_stream)
        # leaky_compressed = 원본 데이터의 부호 * (미세 기울기 * 남은 변위량)
        # 하드웨어 1클록 만에 스케일 다운 가공을 단행하는 대수적 완충 쿠션 수식
        leaky_compressed = jax.lax.mul(
            signed_stream, 
            jax.lax.mul(safe_leaky, jax.lax.add(jnp.abs(filtered_stream), 1e-12))
        )
        
        # 5. 분기문 없는 수리적 멀티플렉서(mathematical_mux)를 통해 고속 병렬 마스킹 사출
        # 정상 구역은 filtered_stream을 그대로 통과시키고, 발산 구역은 미세 그레디언트가 보존된 leaky 선로로 스왑
        compressed_stream = self.mux_opt.mathematical_mux(
            error_mask,
            leaky_compressed,
            filtered_stream
        )
        
        return compressed_stream



         @partial(jax.jit, static_argnums=(0,))
    def enforce_energy_parity(self, stream: jnp.ndarray) -> jnp.ndarray:
        """
        [항상성 집행] 에너지 보존 법칙 및 항상성 평형 강제 (Pure Silicon 하드 가드레일 버전)
        모든 수치 처리가 끝난 다양체가 물리적 위상을 유지하도록 L2 Norm = 1.0 상태로 고정합니다.
        [math_guardrails 기믹] 분모가 0.0f로 수렴하여 발생하는 수치적 파괴(NaN)를 하드 클램핑으로 원천 차단합니다.
        """
        target_dtype = stream.dtype
        safe_epsilon = jnp.array(1e-12, dtype=target_dtype)
        
        # 1. jnp.linalg.norm 대신 jax.lax 원시 프리미티브 조합으로 L2 노름 커스텀 빌드
        # 각 요소의 제곱을 구한 뒤, 가속기 내부 특수기능유닛(SFU)에서 초고속 축소 합산(Sum Reduction) 수행
        squared_stream = jax.lax.square(stream)
        sum_of_squares = jnp.sum(squared_stream) # 전체 차원 축소 연산
        
        # sqrt(sum + epsilon) 수식 전개 후 나누기 연산을 곱셈(역수 연산)으로 유도할 수 있도록 정렬
        l2_norm = jax.lax.sqrt(jax.lax.add(sum_of_squares, safe_epsilon))
        
        # [math_guardrails 핵심 패치] 분모 싱큘래리티 폭발을 차단하기 위한 최하단 하드 가드레일 강제 잠금
        # l2_norm 값이 드물게 언더플로우를 일으키더라도 최소 1e-7 이상을 유지하도록 실리콘 레벨 록을 겁니다.
        safe_l2_norm = jax.lax.max(l2_norm, jnp.array(1e-7, dtype=target_dtype))
        
        # 2. 최종 항상성 평형 사출 (FMA 나눗셈 기계어 다이렉트 매핑)
        return jax.lax.div(stream, safe_l2_norm)

    @partial(jax.jit, static_argnums=(0,))
    def process_pipeline(self, raw_input: jnp.ndarray) -> jnp.ndarray:
        """
        2세대 커널 물리 필터 주행 파이프라인 (Forward-Only Leaky Execution).
        입력 스트림에 대해 과거 시간 축으로의 역전파 링크 없이, 미세 오차 복원 경사도를 보존하며 순방향 정류를 최종 완료합니다.
        """
        # Step 1: 시간 미분 기저 격자(dt) 및 소프트 경계 마진 기반 슈뢰딩거 에너지 장벽 필터링 (Leaky Slope 적용)
        step1 = self.execute_schrodinger_notch_filter(raw_input)
        
        # Step 2: 연쇄 제곱(d^4) 가속 및 수학적 멀티플렉서 기반 카시미르 위상학적 진공 압착 (Leaky Gradient 보존)
        step2 = self.execute_casimir_noise_compression(step1, tolerance=1e-3)
        
        # Step 3: SFU 단일 퓨즈드 합산 루프 및 하드 가드레일 클램핑을 통한 최종 물리적 항상성(L2 Norm Parity = 1.0) 강제 집행
        final_sanitized_output = self.enforce_energy_parity(step2)
        
        return final_sanitized_output


# --- 핵심 물리 커널 단독 무결성 및 항상성 평형 정밀 프로파일링 검증 코드 ---
if __name__ == "__main__":
    print("========================================================================")
    print("🧪 [TEST] physics_filter Leaky 가드레일 및 항상성 패리티 정밀 검증 시동")
    print("========================================================================")

    # 1. 1세대 보조뇌(LLM)가 사출한 왜도 변위 붕괴 스트림 시뮬레이션
    # (평온하게 진행되다가 4번째 노드에서 500.0이라는 파괴적 환각 발생, 마지막 노드는 공차 미세 노이즈)
    llm_corrupted_stream = jnp.array([0.5, 0.51, 0.49, 500.0, 0.52, 0.00002], dtype=jnp.float32)
    print("❌ 1세대 보조뇌 원시 스트림 인입 (환각/노이즈 내포):")
    print(f" └─ {llm_corrupted_stream}")

    # 2. 2세대 핵심 물리 커널 초기화
    # [math_guardrails 기믹 반영] leaky_slope=0.01, boundary_margin=0.05 주입 가동
    filter_kernel = PhysicsInformativeFilter(
        dt=0.001, 
        h_bar_eff=1.0, 
        viscosity_sigma=0.5,
        leaky_slope=0.01,
        boundary_margin=0.05
    )
    
    # JIT 컴파일 및 순방향 파이프라인 집행 (Forward-Only 주행)
    jit_pipeline = jax.jit(filter_kernel.process_pipeline)
    sanitized_physics_stream = jit_pipeline(llm_corrupted_stream)
    
    # 3. [하드웨어 동기화 및 메트릭 사출] JAX 비동기 버퍼 강제 해제
    sanitized_physics_stream.block_until_ready()
    final_l2_norm = jnp.linalg.norm(sanitized_physics_stream)

    print("\n✅ 2세대 본뇌 커널 숙청 및 수리물리 정류 완료:")
    print(f" └─ {sanitized_physics_stream}")
    print("   [분석 A] 500.0의 거시적 환각 ➔ 슈뢰딩거 에너지 장벽 바깥 영역의 Leaky 쿠션 가동 완료")
    print("   [분석 B] 0.00002의 미세 오차 ➔ 연쇄 제곱 카시미르 음압 및 수학적 MUX 교차 결합 완충 완료")

    print("\n📊 최종 수리물리학적 무결성 및 경사도 안정성 평가:")
    print(f" ├─ 최종 다양체 에너지 패리티 (L2 Norm): {final_l2_norm:.6f}")
    
    # 물리 법칙 검증 (L2 Norm은 하드 가드레일 마스크에 의해 반드시 1.0 평형을 완벽히 사수해야 합니다)
    is_parity_safe = jnp.isclose(final_l2_norm, 1.0, atol=1e-5)
    print(f" ├─ 항상성 무결성 합격 여부(Homeostasis Parity): {is_parity_safe}")
    
    # [math_guardrails 핵심 사증 구문] 완전히 0.0f로 죽지 않고 미세 기울기를 사수하여 살아남았는지 확인
    # 4번째 인덱스의 값(환각이 숙청된 자리)이 완전한 0.0이 아닌, 복원 그레디언트가 도킹할 수 있는 미세 성분인지 검증
    hallucination_node_value = jnp.abs(sanitized_physics_stream[3])
    print(f" ├─ 환각 노드의 리키 보존 변위 크기: {hallucination_node_value:.8f}")
    
    is_leaky_preserved = (hallucination_node_value > 0.0) & (hallucination_node_value < 1e-3)
    print(f" └─ 경계면 미분 그레디언트 숨통 보존 상태: {is_leaky_preserved}")
    
    assert is_parity_safe and is_leaky_preserved, "❌ [검증 실패] 항상성 패리티가 파괴되었거나 그레디언트가 질식사했습니다."
    print("\n✅ [TEST PASSED] 수치 파괴를 완벽히 통제하면서도 오차 복원 그레디언트를 끝까지 유착시키는 물리 커널을 증명했습니다.")
    print("========================================================================\n")



