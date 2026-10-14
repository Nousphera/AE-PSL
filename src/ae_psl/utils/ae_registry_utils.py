import hashlib
import glob
import json
import os

from ae_psl.utils.file_utils import load_object_from_json, save_object_to_json
import torch





# ---- AE Storage and Retrieval ----
def prepare_ae_dir(signature):
    """
    Prepares a directory to save AE weights and signature.
    Returns the path to the directory.
    """
    checkpoint_folder_path = os.environ.get('AE_WEIGHTS_DIR')

    save_dir_name = filename_from_signature(signature)

    save_path = os.path.join(checkpoint_folder_path, save_dir_name)
    os.makedirs(save_path, exist_ok=True)

    return save_path

def save_ae_with_signature(model, signature):
    """
    Saves the AE model weights and a corresponding signature.json file.
    Use this function at the end of your pre-training script.
    """
    checkpoint_folder_path = os.environ.get('AE_WEIGHTS_DIR')

    save_dir_name = filename_from_signature(signature)

    save_path = os.path.join(checkpoint_folder_path, save_dir_name)
    os.makedirs(save_path, exist_ok=True)

    # 1. Save Weights
    torch.save(model.state_dict(), os.path.join(save_path, 'ae_model.pth'))

    save_object_to_json(signature, os.path.join(save_path, 'signature.json'))
    print(f"Saved AE and signature to {save_path}")


def verify_signature(signature_in_file, required_params):
    """
    Compares the loaded signature against required parameters.
    Returns True only if ALL required keys match.
    """
    for key, value in required_params.items():
        # If the key is not in the file, we can't verify it -> Warning
        if key not in signature_in_file:
            return False
        # If values don't match -> Fail
        # Note: Be careful with types (int vs str), might need casting
        if str(signature_in_file[key]) != str(value):
            return False

    return True


def find_matching_pretrained_ae(required_params):
    """
    Scans the AE_CHECKPOINT_DIR for any folder containing a signature.json
    that matches the required_params.
    """
    checkpoint_folder_path = os.environ.get('AE_WEIGHTS_DIR')

    if not os.path.exists(checkpoint_folder_path):
        print(f"Warning: AE directory {checkpoint_folder_path} does not exist.")
        return None

    # Get list of all signature files
    # Structure: .../autoencoders/*/signature.json
    signature_files = glob.glob(os.path.join(checkpoint_folder_path, '*', 'signature.json'))

    for sig_file in signature_files:
        try:
            signature = load_object_from_json(sig_file)

            if verify_signature(signature, required_params):
                # Found a match! Return the sibling model file
                model_dir = os.path.dirname(sig_file)
                model_path = os.path.join(model_dir, 'ae_model.pth')
                if os.path.exists(model_path):
                    print(f"Found matching AE at: {model_path}")
                    return model_dir
        except Exception as e:
            print(f"Error reading signature {sig_file}: {e}")
            continue

    print("No matching pre-trained AE found.")
    return None


def load_auto_encoder_model(global_args, model, signature, device):
    """
    The main entry point for loading an AE.
    """

    weights_path = None

    if global_args['ae_specific_weights_path']:
        weights_path = global_args['ae_specific_weights_path']
        weights_dir = os.path.dirname(weights_path)
    else:
        # Define what constitutes a "match" for this experiment
        # required_params = {
        #     'type': global_args['ae_type'],
        #     'dataset': global_args['ae_pretrain_dataset'],
        #     'dataset_proportion': global_args['ae_pretrain_dataset_fraction'],
        #     'model': global_args['model'],
        #     'split_layer': global_args['split_layer'],
        #     'latent_dim': global_args['ae_latent_dim'],
        # }
        # we the required params to the entire signature, however sometimes we might not care about certain parameters to match.
        required_params = signature
        weights_dir = find_matching_pretrained_ae(required_params)



    # 3. Load weights if found
    if weights_dir:
        weights_path = weights_path or os.path.join(weights_dir, 'ae_model.pth')
        try:
            checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        except Exception as e:
            raise RuntimeError(f"Error loading AE weights from {weights_path}: {e}")

        model.load_state_dict(checkpoint)
        if global_args['ae_specific_weights_path']:
            print(f"Successfully loaded AE weights from specific weight path: {weights_path}")
        else:
            print("Successfully loaded AE weights from previously trained matching pre-trained AE.")
    else:
        return None, None

    weights_path = os.path.join(weights_dir, 'ae_model.pth')
    results_path = os.path.join(weights_dir, 'results.json')

    results = {}
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    results['reused_AE'] = True

    return model, results




def filename_from_signature(signature: dict) -> str:
    """Name the cache by every declared input, never a truncated partial key."""
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    return "ae-" + hashlib.sha256(encoded).hexdigest()
