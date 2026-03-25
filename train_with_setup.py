#!/usr/bin/env python
"""
Wrapper script that sets up environment and runs training.
Ensures gymnasium->gym compatibility before importing training code.
"""

import sys
import _setup_env  # Load gymnasium->gym compatibility shim first

# Now run the actual training script
if __name__ == "__main__":
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    import hydra
    from omegaconf import OmegaConf
    import pathlib

    # Allow eval in configs
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    @hydra.main(
        version_base=None,
        config_path=str(pathlib.Path(__file__).parent.joinpath(
            'diffusion_policy','config'))
    )
    def main(cfg: OmegaConf):
        # Resolve immediately so all resolvers use the same time
        OmegaConf.resolve(cfg)

        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.run()

    main()
