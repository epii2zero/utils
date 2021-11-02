import torch

def show_model_structure(model_odict):
    """
    show model names and its structure
    input: ordered dictionary containing models
    """
    for name, model in model_odict.items():
        num_total_params = sum(p.numel() for p in model.parameters())
        print(f'model name: {name}\n'
              f'The number of parameters: {num_total_params:,}\n'
              f'model structure: {model}\n')
    return

def test_model_operation(model_odict, test_inputs):
    model_list = model_odict.values()
    test_outputs = []
    if len(model_list) != len(test_inputs):
        raise ValueError('The number of models and its test inputs must be same')
    for test_model, test_input in zip(model_list, test_inputs):
        output = test_model(test_input)
        test_outputs.append(output)
    return test_outputs

def _unpacking(datum):
    """
        Unpack the tuple packaged datum
    """
    if type(datum) != tuple: # Only for tuple
        return datum
    ret = []
    for data in datum:
        if type(data) == tuple:
            for element in data:
                ret.append(element)
        else:
            ret.append(data)
    return tuple(ret)

def _show_data(datum, mode):
    """
        Show the torch shapes of the data in datum
    """
    if type(datum) == tuple:
        for i, data in enumerate(datum):
            if type(data) == list:
                print(f'test {mode}[{i}] is the list:')
                for j, sub_data in enumerate(data):
                    if type(sub_data) == torch.Tensor:
                        print(f'  test {mode}[{i}][{j}] shape: {sub_data.shape}')
                    else:
                        print(f'  type of test {mode}[{i}][{j}] is not intended')
            elif type(data) == torch.Tensor:
                print(f'test {mode}[{i}] shape: {data.shape}')
            else:
                print(f'type of test {mode}[{i}] is not inteded')
    elif type(datum) == torch.Tensor:
        print(f'test {mode} shape: {datum.shape}')
    else:
        print(f'type of test {mode} is not inteded')
            
def show_testIO_shape(test_inputs, test_outputs):
    """
        Show the torch shapes of the inputs and outputs of model
    """
    for test_input, test_output in zip(test_inputs, test_outputs):
        # Unfold packaging (inner tuple)
        unpacked_input = _unpacking(test_input)
        unpacked_output = _unpacking(test_output)
        # Describe datum
        _show_data(datum=unpacked_input, mode='input')
        _show_data(datum=unpacked_output, mode='output')