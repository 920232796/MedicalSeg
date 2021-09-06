


def average_metric(metric_res):
    return_res = metric_res[0]
    total = len(metric_res)
    for m_res in metric_res[1:]:
        for m_name, m_value in m_res.items():
            return_res[m_name] = return_res[m_name] + m_value


    for m_name, _ in return_res.items():
            return_res[m_name] = return_res[m_name] / total
    return return_res