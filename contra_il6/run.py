import click
from .inference import CONTRA_IL6_Inferencer, CONTRA_IL6_Predictor
import yaml
from os.path import dirname, abspath, join, exists
from .utils import download_ckpts

__CURRENT_DIR__ = dirname(abspath(__file__))
__CKPT_URL__ = "https://huggingface.co/cbbl-skku-org/CONTRA-IL6/resolve/main/ckpts.zip"
__CKPT_DIR__ = join(__CURRENT_DIR__, 'ckpts')
__MODEL_CONFIG__ = yaml.safe_load(open(join(__CURRENT_DIR__, 'default_config.yaml')))['model']

# Check if there is no checkpoint directory, download and extract it
if not exists(__CKPT_DIR__):
    print("Checkpoint directory not found. Downloading and extracting checkpoints...")
    download_ckpts(__CKPT_URL__, __CURRENT_DIR__)


predictor = CONTRA_IL6_Predictor(__MODEL_CONFIG__, __CKPT_DIR__, 10, 'cpu')
inferencer = CONTRA_IL6_Inferencer(predictor, 'cpu')

@click.command()
@click.option('--input', '-i', type=click.Path(exists=True), help='Path to a FASTA file containing peptide sequences.')
@click.option('--threshold', '-t', default=0.46, help='Threshold for classification (default: 0.46).')
@click.option('--batch_size', '-b', default=4, help='Batch size for processing (default: 4).')
@click.option('--output', '-o', type=click.Path(), help='Output file to save predictions.')
def cli(input, threshold, batch_size, output):
    input_dict = inferencer.read_fasta_file(input)
    output_dict = inferencer.predict_sequences(input_dict, threshold=threshold, batch_size=batch_size)
    inferencer.write_output_file(output_dict, output)
    print(f"Predictions saved to {output}")
    
if __name__ == "__main__":
    cli()