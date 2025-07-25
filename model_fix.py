import h5py
import json

def fix_keras_model(model_path, output_path):
    """
    Fix Keras model by removing unsupported 'groups' parameter from DepthwiseConv2D layers
    """
    with h5py.File(model_path, 'r+') as f:
        # Read model config
        model_config = f.attrs['model_config']
        if isinstance(model_config, bytes):
            model_config = model_config.decode('utf-8')
        
        config_dict = json.loads(model_config)
        
        # Function to recursively fix layers
        def fix_layers(layers):
            for layer in layers:
                if layer.get('class_name') == 'DepthwiseConv2D':
                    config = layer.get('config', {})
                    if 'groups' in config:
                        print(f"Removing 'groups' parameter from layer: {config.get('name', 'unnamed')}")
                        del config['groups']
                
                # Handle nested models
                if 'layers' in layer.get('config', {}):
                    fix_layers(layer['config']['layers'])
                elif 'config' in layer and isinstance(layer['config'], dict) and 'layers' in layer['config']:
                    fix_layers(layer['config']['layers'])
        
        # Fix the model
        if 'config' in config_dict and 'layers' in config_dict['config']:
            fix_layers(config_dict['config']['layers'])
        elif 'layers' in config_dict:
            fix_layers(config_dict['layers'])
        
        # Write back the fixed config
        f.attrs['model_config'] = json.dumps(config_dict).encode('utf-8')
    
    print(f"Model fixed and saved to: {output_path}")

# Usage
if __name__ == "__main__":
    fix_keras_model("keras_model.h5", "keras_model_fixed.h5")