
import numpy as np

def average_metric(metric_res):
    return_res = metric_res[0]
    total = len(metric_res)
    average_num = { m_name: total for m_name, _ in metric_res[0].items()}
    return_res =  { m_name: 0.0 for m_name, _ in metric_res[0].items()}

    for m_res in metric_res:
        for m_name, m_value in m_res.items():
            if np.isnan(m_value): #发现nan，所以num - 1
                m_value = 0.0
                average_num[m_name] -= 1

            return_res[m_name] = return_res[m_name] + m_value


    for m_name, _ in return_res.items():
            if average_num[m_name] == 0:
                return_res[m_name] = 0.0
                continue
            return_res[m_name] = return_res[m_name] / average_num[m_name] 
    return return_res


if __name__ == "__main__":
    metric_res = [{"dice_1": float("NaN"), "dice_2": 0.1}, {"dice_1": 0.5, "dice_2": float("NaN")}, {"dice_1": float("NaN"), "dice_2": 0.3}]   
    print(average_metric(metric_res))