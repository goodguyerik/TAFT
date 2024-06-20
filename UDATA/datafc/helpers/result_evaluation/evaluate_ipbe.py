from pathlib import Path

from sacred import Experiment
from sacred.observers import MongoObserver
import ujson as json

ipbe_result_data_path = Path(
    r"C:\Users\Clapika\Projects\ipbe\DataCleaning\results\ipbe"
)
flashfill_result_data_path = Path(
    r"C:\Users\Clapika\Projects\ipbe\DataCleaning\results\flashfill"
)

ex = Experiment("jupyter_ex", interactive=True)
ex.observers.append(MongoObserver.create())


@ex.main
def record(dataset, mapping_method):
    if mapping_method == "ipbe":
        result_data_path = ipbe_result_data_path
    else:
        result_data_path = flashfill_result_data_path
    output_file = result_data_path / f"{dataset}.json"
    json_obj = json.load(output_file.open("r"))
    return json_obj


for ex_dataset in ["ijcai", "sygus", "museum", "nyc", "prog"]:
    ex.run(config_updates={"dataset": ex_dataset, "mapping_method": "ipbe"})
