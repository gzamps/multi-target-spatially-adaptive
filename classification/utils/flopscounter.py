import numpy as np
import torch
import torch.nn as nn

def flops_to_string(flops):
    if flops // 10**9 > 0:
        return str(round(flops / 10.**9, 2)) + 'GMac'
    elif flops // 10**6 > 0:
        return str(round(flops / 10.**6, 2)) + 'MMac'
    elif flops // 10**3 > 0:
        return str(round(flops / 10.**3, 2)) + 'KMac'
    return str(flops) + 'Mac'

def get_model_parameters_number(model, as_string=True):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not as_string:
        return params_num

    if params_num // 10 ** 6 > 0:
        return str(round(params_num / 10 ** 6, 2)) + 'M'
    elif params_num // 10 ** 3:
        return str(round(params_num / 10 ** 3, 2)) + 'k'

    return str(params_num)

def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)
    net_main_module.compute_total_flops_cost = compute_total_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    # Adding variables necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__
            # print ( module , "    ", module.__flops__)
            # print ( module , " PERBATCH   ", module.__flops__/batches_count)

        else:
            # print("not supported: ", module)
            pass

    # print( "FLOPS SUM ", flops_sum)
    # print( "BATCHES COUNT ", batches_count)
    # print( "AVE FLOPS: ", flops_sum/batches_count)

    return flops_sum / batches_count, flops_sum, batches_count


def compute_total_flops_cost(self):
    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__
    return flops_sum, batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):
    def add_flops_mask_func(module):
        if isinstance(module, torch.nn.Conv2d):
            module.__mask__ = mask
    module.apply(add_flops_mask_func)


def remove_flops_mask(module):
    module.apply(add_flops_mask_variable_or_reset)

# ---- Internal functions
def is_supported_instance(module):
    mode = 'basic'
    mode = 'all'

    if mode == 'basic': # only convs
        if isinstance(module, (torch.nn.Conv2d,)):
            return True
    else:
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ReLU, torch.nn.PReLU, torch.nn.ELU, \
                            torch.nn.LeakyReLU, torch.nn.ReLU6, torch.nn.Linear, \
                            torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.BatchNorm2d, \
                            torch.nn.Upsample, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
            return True
    return False


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    assert isinstance(input, tuple)
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += output_elements_count


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    if module.__mask__ is not None:
        active_elements_count *= float(module.__mask__.active_positions / module.__mask__.total_positions)
    module.__flops__ += active_elements_count


def linear_flops_counter_hook(module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    # print( "In LIONEAR HOOK")        
    batch_size = input.shape[0]
    # module.__flops__ += batch_size * input.shape[1] * output.shape[1]
    # print( input.shape)
    # print ( "in: ", input.shape)
    # print ( "out: ", output.shape)
    # print("")



    if len(input.shape) == 4:
        if module.__mask__ is None:
            active_elements_count = batch_size * input.shape[1] * input.shape[2] #dense
        else: 
            active_elements_count = float(module.__mask__.active_positions) #sparse

        # module.__flops__ += batch_size * input.shape[1] * input.shape[2] * input.shape[3] *  output.shape[1]  + batch_size * input.shape[1] * input.shape[2] * output.shape[1] # Convnext linear piecewise
        # module.__flops__ += batch_size * input.shape[1] * input.shape[2] * input.shape[3] *  output.shape[1]  * 2 # Convnext linear piecewise
        # module.__flops__ += batch_size * input.shape[1] * input.shape[2] * output.shape[3] *( 2 * input.shape[3] - 1) # Convnext linear piecewise + bias?

        # WORKS
        # module.__flops__ += batch_size * input.shape[1] * input.shape[2] * output.shape[3] * input.shape[3]  # Convnext linear piecewise, works WORKS FOR DENSE
        overall_conv_flops = output.shape[3] * input.shape[3] * active_elements_count # Cin* Cout * HW
        

        module.__flops__ += overall_conv_flops

    else:
        module.__flops__ += batch_size * input.shape[1] * output.shape[1] # normal linears


    # print( module, "  " , module.__flops__)



def pool_flops_counter_hook(module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    module.__flops__ += np.prod(input.shape)

def bn_flops_counter_hook(module, input, output):
    if isinstance(input, tuple):
        input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    if module.__mask__ is not None:
        batch_flops *= float(module.__mask__.active_positions / module.__mask__.total_positions)
    module.__flops__ += batch_flops

def conv_flops_counter_hook(conv_module, input, output):
    if isinstance(input, tuple):
        input = input[0]
    # print( "In CONV HOOK")

    batch_size, _, output_height, output_width = output.shape

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel


    if conv_module.__mask__ is None:
        active_elements_count = batch_size * output_height * output_width
    else:
        active_elements_count = float(conv_module.__mask__.active_positions)


    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    # print( conv_module, "  " , overall_flops)
    conv_module.__flops__ += overall_flops

def batch_counter_hook(module, input, output):
    # Can have multiple inputs, getting the first one
    if isinstance(input, tuple):
        input = input[0]
    batch_size = input.shape[0]
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return
        if isinstance(module, torch.nn.Conv2d):
            function = (conv_flops_counter_hook)
        elif isinstance(module, (torch.nn.ReLU, torch.nn.PReLU, torch.nn.ELU, \
                                 torch.nn.LeakyReLU, torch.nn.ReLU6)):
            function = (relu_flops_counter_hook)
        elif isinstance(module, torch.nn.Linear):
            function = (linear_flops_counter_hook)
        elif isinstance(module, (torch.nn.AvgPool2d, torch.nn.MaxPool2d, nn.AdaptiveMaxPool2d, \
                                 nn.AdaptiveAvgPool2d)):
            function = (pool_flops_counter_hook)
        elif isinstance(module, torch.nn.BatchNorm2d):
            function = (bn_flops_counter_hook)
        elif isinstance(module, torch.nn.Upsample):
            function = (upsample_flops_counter_hook)
        else:
            function = (empty_flops_counter_hook)
        
        handle = module.register_forward_hook(function)
        module.__flops_function__ = function
        module.__flops_handle__ = handle

def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__
            del module.__flops_function__

# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
    if is_supported_instance(module):
        module.__mask__ = None
    #     print ( "Suppoerted ", module)
    # else:
    #     print ( " not Suppoerted")
