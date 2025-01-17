from google.cloud import aiplatform

def train_model():
    aiplatform.init(project='mlopssentimentanalysis', location='europe-west1', staging_bucket='gs://your-staging-bucket')

    job = aiplatform.CustomJob.from_local_script(
        display_name='sentiment-analysis-training',
        script_path='src/sentiment_analysis/train.py',
        container_uri='gcr.io/deeplearning-platform-release/tf2-cpu.2-3:latest',
        requirements=['torch', 'transformers', 'dvc'],
        args=['train', './data/processed', '--config-name', 'config.yaml'],
        replica_count=1,
        machine_type='n1-standard-4',
    )

    job.run()

if __name__ == "__main__":
    train_model()