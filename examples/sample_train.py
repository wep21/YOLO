import sys
from pathlib import Path

import hydra

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import Config and the new TrainModel
from yolo.config.config import Config # Assuming Config is here
from yolo.tools.solver import TrainModel
# Removed ModelTrainer, ProgressLogger, create_converter, create_dataloader, create_model, get_device


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: Config):
    # Instantiate TrainModel. It handles its own internal setup including model creation,
    # data loaders, device handling, etc., based on the config.

    # The TrainModel's __init__ method already creates the model, data loaders,
    # optimizer, scheduler, loss function, and moves model to device.
    train_model = TrainModel(cfg)

    # Determine the number of epochs from config.
    # The old ModelTrainer didn't explicitly take epochs in solve(), it was likely part of cfg.task.epoch
    epochs = getattr(cfg.task, "epoch", 1) # Default to 1 epoch if not specified in cfg

    # Start the training process using the new train_loop
    # train_model.model.to(train_model.device) # TrainModel's __init__ should handle this.
                                            # Explicit call train_model.to(train_model.device) if TrainModel itself is a module
                                            # and needs to be moved. But BaseModel (parent of TrainModel) handles device for self.model

    train_model.train_loop(epochs=epochs)

    # ProgressLogger, custom device handling (use_ddp), direct dataloader/model creation,
    # converter creation, and the old ModelTrainer.solve() are replaced by TrainModel's encapsulated logic.

if __name__ == "__main__":
    main()
