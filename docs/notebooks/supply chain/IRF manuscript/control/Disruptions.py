# cap_dis_t_idx = 40  # Disruption starts here
# cap_early_warning = 1  # Information about the disruption is relayed to the model
# capacity_factor_dict = {
#     k: [1] * (cap_dis_t_idx - 1) + [int(i < cap_dis_t_idx - cap_early_warning - 1)] * (
#                 schedule_exec_scenarios - cap_dis_t_idx + 1) for i in range(schedule_exec_scenarios)
#     for k in range(i * schedule_time_intervals, (i + 1) * schedule_time_intervals)
# }

# truck_dis_start_idx = 27
# truck_dis_end_idx = 39
# truck_early_warning = 2
#
# prefix = [1] * (truck_dis_start_idx - 1)
# normal_vec = prefix + [1] * (schedule_exec_scenarios - truck_dis_start_idx + 1)
# drop_vec = prefix + [0.75] * (truck_dis_end_idx - truck_dis_start_idx + 1) \
#            + [1] * (schedule_exec_scenarios - truck_dis_end_idx)
#
# threshold = truck_dis_start_idx - truck_early_warning - 1
#
# truck_factor_dict = {
#     k: (normal_vec if i < threshold else drop_vec)
#     for i in range(schedule_exec_scenarios)
#     for k in range(i * schedule_time_intervals, (i + 1) * schedule_time_intervals)
# }
#
# resource_dis_start_idx = 27
# resource_dis_end_idx = 39
# resource_early_warning = 2
#
# prefix = [1] * (resource_dis_start_idx -1)
# normal_vec = prefix + [1] * (schedule_exec_scenarios - resource_dis_start_idx + 1)
# drop_vec = prefix + [0] * (resource_dis_end_idx - resource_dis_start_idx + 1) \
#            + [1] * (schedule_exec_scenarios - resource_dis_end_idx)
#
# threshold = resource_dis_start_idx - resource_early_warning - 1
#
# resource_factor_dict = {
#     k: (normal_vec if i < threshold else drop_vec)
#     for i in range(schedule_exec_scenarios)
#     for k in range(i * schedule_time_intervals, (i + 1) * schedule_time_intervals)
# }
#
# disruption_dict = dict()
# disruption_dict[('loc2', 'com1_process')] = capacity_factor_dict
# disruption_dict[('truck45', 'com1_loc4_out')] = truck_factor_dict
# disruption_dict[('loc1', 'com1_pur')] = resource_factor_dict