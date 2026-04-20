"""
Pipeline Kubeflow KFP v2 — MLOps GPON/DSL Diagnostic
"""

from kfp import dsl

REGISTRY         = "localhost:5000"
MINIO_ENDPOINT   = "http://10.98.20.211:9000"
MINIO_ACCESS_KEY = "minio"
MINIO_SECRET_KEY = "minio123"
OLLAMA_BASE_URL  = "http://localhost:11434"


def set_minio_env(step):
    step.set_env_variable("MINIO_ENDPOINT",   MINIO_ENDPOINT)
    step.set_env_variable("MINIO_ACCESS_KEY", MINIO_ACCESS_KEY)
    step.set_env_variable("MINIO_SECRET_KEY", MINIO_SECRET_KEY)
    return step


def set_minio_ollama_env(step):
    set_minio_env(step)
    step.set_env_variable("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    return step


# ── COMPOSANTS HN ──

@dsl.container_component
def preprocessing_hn():
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/preprocessing:v1",
        command=["python"],
        args=["preprocess_huawei_nokia.py"],
    )

@dsl.container_component
def healthscore_hn():
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/healthscore:v1",
        command=["bash", "-c"],
        args=["python train_huawei_nokia.py && python evaluate_huawei_nokia.py"],
    )

@dsl.container_component
def root_cause_hn():
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/root-cause:v1",
        command=["bash", "-c"],
        args=["python build_faiss_index_hn.py && python main_hn.py"],
    )

@dsl.container_component
def prediction_proactive_hn():
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/prediction-proactive:v1",
        command=["bash", "-c"],
        args=["python preprocess_hn_scenario3.py && python scenario3_hn.py"],
    )


# ── COMPOSANTS IGD ──

@dsl.container_component
def preprocessing_igd():
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/preprocessing:v1",
        command=["python"],
        args=["preprocess_igd.py"],
    )

@dsl.container_component
def healthscore_igd():
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/healthscore:v1",
        command=["bash", "-c"],
        args=["python train_igd.py && python evaluate_igd.py"],
    )

@dsl.container_component
def root_cause_igd():
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/root-cause:v1",
        command=["bash", "-c"],
        args=["python build_faiss_index_igd.py && python main_igd.py"],
    )

@dsl.container_component
def prediction_proactive_igd():
    return dsl.ContainerSpec(
        image=f"{REGISTRY}/prediction-proactive:v1",
        command=["bash", "-c"],
        args=["python preprocess_igd_scenario3.py && python scenario3_igd.py"],
    )


# ── PIPELINE PRINCIPAL ──

@dsl.pipeline(
    name="mlops-gpon-dsl-diagnostic",
    description="Pipeline MLOps complet — GPON/DSL Diagnostic",
)
def mlops_diagnostic_pipeline():

    # BRANCHE HN
    step_preprocess_hn = preprocessing_hn()
    step_preprocess_hn.set_display_name("Preprocessing — Huawei/Nokia")
    step_preprocess_hn.set_retry(num_retries=2)
    set_minio_env(step_preprocess_hn)

    step_healthscore_hn = healthscore_hn()
    step_healthscore_hn.set_display_name("HealthScore + Gatekeeping — HN")
    step_healthscore_hn.after(step_preprocess_hn)
    step_healthscore_hn.set_retry(num_retries=1)
    set_minio_env(step_healthscore_hn)

    step_root_cause_hn = root_cause_hn()
    step_root_cause_hn.set_display_name("Root Cause RAG — HN")
    step_root_cause_hn.after(step_healthscore_hn)
    set_minio_ollama_env(step_root_cause_hn)

    step_prediction_hn = prediction_proactive_hn()
    step_prediction_hn.set_display_name("Prediction Proactive — HN")
    step_prediction_hn.after(step_root_cause_hn)
    set_minio_env(step_prediction_hn)

    # BRANCHE IGD
    step_preprocess_igd = preprocessing_igd()
    step_preprocess_igd.set_display_name("Preprocessing — IGD")
    step_preprocess_igd.set_retry(num_retries=2)
    set_minio_env(step_preprocess_igd)

    step_healthscore_igd = healthscore_igd()
    step_healthscore_igd.set_display_name("HealthScore + Gatekeeping — IGD")
    step_healthscore_igd.after(step_preprocess_igd)
    step_healthscore_igd.set_retry(num_retries=1)
    set_minio_env(step_healthscore_igd)

    step_root_cause_igd = root_cause_igd()
    step_root_cause_igd.set_display_name("Root Cause RAG — IGD")
    step_root_cause_igd.after(step_healthscore_igd)
    set_minio_ollama_env(step_root_cause_igd)

    step_prediction_igd = prediction_proactive_igd()
    step_prediction_igd.set_display_name("Prediction Proactive — IGD")
    step_prediction_igd.after(step_root_cause_igd)
    set_minio_env(step_prediction_igd)

    # LIMITES RESSOURCES
    for step in [
        step_preprocess_hn, step_healthscore_hn, step_root_cause_hn, step_prediction_hn,
        step_preprocess_igd, step_healthscore_igd, step_root_cause_igd, step_prediction_igd,
    ]:
        step.set_cpu_limit("1")
        step.set_memory_limit("2G")


if __name__ == "__main__":
    from kfp import compiler
    output_file = "mlops_diagnostic_pipeline.yaml"
    compiler.Compiler().compile(
        pipeline_func=mlops_diagnostic_pipeline,
        package_path=output_file,
    )
    print(f"✅ Pipeline compilé → {output_file}")