# region_model_configs.py

from configs.panoptic_model_configs import RegionCellCombination, MuTILsParams, VisConfigs

# 本文件可能作为默认配置文件使用，提供默认的 MuTILsParams 实例或其他扩展设置。
# 假设默认情况下直接使用 panoptic_model_configs 中定义的参数类作为配置。

# 为了与通过 load_configs 函数的用法保持一致，我们可以直接提供类或实例。
# 若您的代码在其他文件中期望通过 cfg.MuTILsParams 访问配置，那么这里保持类的形式即可。
# 若期望通过 cfg 本身作为实例访问，可将 MuTILsParams 实例化并赋值给 cfg。

# 简单用法：直接将 MuTILsParams 作为 cfg 暴露（类本身）
# cfg = MuTILsParams

# 如果需要一个已经实例化的配置对象，可以这样：
cfg = MuTILsParams

# 如果需要对参数进行一定修改（如覆盖某些默认参数），可以在实例化后修改属性。例如：
# new_params = MuTILsParams
# new_params.model_params['roi_side'] = 512  # 假设你想修改参数
# cfg = new_params

# 若只需要导出类和参数供外部使用，则可不实例化，仅做透传
# 例如直接将类透传：
# class cfg:
#     RegionCellCombination = RegionCellCombination
#     MuTILsParams = MuTILsParams
#     VisConfigs = VisConfigs

# 具体选择依您的项目代码在其他地方是如何使用 cfg 的属性。
