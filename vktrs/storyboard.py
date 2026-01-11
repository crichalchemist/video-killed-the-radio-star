"""
Storyboard management for VKTRS.

The storyboard is the central data structure that stores all project state:
- Scenes with timing and prompts
- Themes (musical groupings)
- Animation parameters
- Audioreactive signals

The storyboard is saved as YAML for human readability and portability.
"""

from pathlib import Path
from typing import Dict, Optional, Any
import logging
import yaml

logger = logging.getLogger(__name__)

try:
    from omegaconf import OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False


# ============================================================================
# Storyboard Class
# ============================================================================

class Storyboard:
    """
    VKTRS storyboard with validation and persistence.

    The storyboard captures all project state in a single YAML file,
    enabling iterative development and portability to other tools.
    """

    def __init__(self, config_path: Optional[Path] = None, config_dict: Optional[Dict] = None):
        """
        Initialize storyboard from file, dict, or create new.

        Args:
            config_path: Path to storyboard.yaml (None = create new)
            config_dict: Dict to initialize from (takes precedence over file)
            
        Raises:
            FileNotFoundError: If config_path is provided but doesn't exist
        """
        if config_dict:
            if OMEGACONF_AVAILABLE:
                self.config = OmegaConf.create(config_dict)
            else:
                self.config = config_dict
            self.path = None
        elif config_path:
            # If path is provided, it must exist
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Storyboard file not found: {config_path}")
            
            if OMEGACONF_AVAILABLE:
                self.config = OmegaConf.load(config_path)
            else:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
            self.path = config_path
        else:
            # No path or dict provided - create new
            self.config = self._create_default_config()
            self.path = None

        self.validate()

    def _create_default_config(self) -> Any:
        """Create default storyboard structure"""
        default = {
            'params': {
                'project_name': None,
                'audio_fpath': None,
                'audio_duration': None,
                'fps': 12,
                'resolution': [512, 512],
                'model': 'runwayml/stable-diffusion-v1-5',
                'use_stability_api': False
            },
            'scenes': [],
            'themes': {},
            'animation_params': {},
            'audioreactive_signals': {}
        }

        if OMEGACONF_AVAILABLE:
            return OmegaConf.create(default)
        else:
            return default

    def validate(self):
        """
        Validate storyboard consistency.

        Checks:
        - First scene starts at t=0
        - Scenes are contiguous (no gaps)
        - No overlapping scenes
        - Last scene matches audio duration

        Raises:
            ValueError: If validation fails
        """
        errors = []

        # Get scenes and params (same access pattern for both OmegaConf and dict)
        scenes = self.config.get('scenes', [])
        params = self.config.get('params', {})

        if not scenes:
            # Empty storyboard is valid
            return

        # First scene must start at 0
        first_start = scenes[0].get('start_time', -1)
        if first_start != 0:
            errors.append(f"First scene must start at t=0 (starts at {first_start})")

        # Scenes must be contiguous
        for i in range(len(scenes) - 1):
            scene_end = scenes[i].get('end_time')
            next_start = scenes[i+1].get('start_time')

            if scene_end is None:
                errors.append(f"Scene {i} missing end_time")
                continue
            if next_start is None:
                errors.append(f"Scene {i+1} missing start_time")
                continue

            if abs(scene_end - next_start) > 0.01:  # 10ms tolerance
                errors.append(
                    f"Gap between scenes {i} and {i+1}: "
                    f"scene {i} ends at {scene_end:.3f}s, "
                    f"scene {i+1} starts at {next_start:.3f}s"
                )
            
            # Check for overlap
            if next_start < scene_end - 0.01:
                errors.append(
                    f"Overlap between scenes {i} and {i+1}: "
                    f"scene {i} ends at {scene_end:.3f}s, "
                    f"scene {i+1} starts at {next_start:.3f}s"
                )

        # Last scene should match audio duration (if known)
        audio_duration = params.get('audio_duration')

        if audio_duration:
            last_end = scenes[-1].get('end_time')
            if last_end and abs(last_end - audio_duration) > 0.1:  # 100ms tolerance
                errors.append(
                    f"Last scene ends at {last_end:.3f}s "
                    f"but audio duration is {audio_duration:.3f}s"
                )

        if errors:
            raise ValueError(
                "Storyboard validation failed:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

    def save(self, path: Optional[Path] = None):
        """
        Save storyboard to YAML file.

        Args:
            path: Output path (uses self.path if not specified)
        """
        path = Path(path) if path else self.path
        if not path:
            raise ValueError("No save path specified")

        path.parent.mkdir(parents=True, exist_ok=True)

        if OMEGACONF_AVAILABLE:
            with open(path, 'w') as f:
                yaml.dump(
                    OmegaConf.to_container(self.config),
                    f,
                    default_flow_style=False,
                    sort_keys=False
                )
        else:
            with open(path, 'w') as f:
                yaml.dump(
                    self.config,
                    f,
                    default_flow_style=False,
                    sort_keys=False
                )

        self.path = path
        logger.info(f"💾 Saved storyboard to {path}")

    def compile(self) -> Dict:
        """
        Compile storyboard for export.

        Resolves all references (themes, signals) and prepares for animation.

        Returns:
            Fully resolved storyboard dict
        """
        if OMEGACONF_AVAILABLE:
            compiled = OmegaConf.to_container(self.config)
        else:
            import copy
            compiled = copy.deepcopy(self.config)

        # Resolve theme references in scenes
        for scene in compiled.get('scenes', []):
            if 'theme' in scene:
                theme_id = scene['theme']
                if theme_id in compiled.get('themes', {}):
                    # Inherit theme properties
                    theme = compiled['themes'][theme_id]
                    for key, value in theme.items():
                        if key not in scene:  # Don't override scene-specific settings
                            scene[key] = value

        return compiled

    def show(self, detailed: bool = False):
        """
        Display storyboard summary.

        Args:
            detailed: Show detailed scene information
        """
        if OMEGACONF_AVAILABLE:
            params = self.config.params
            scenes = self.config.scenes
            themes = self.config.get('themes', {})
        else:
            params = self.config.get('params', {})
            scenes = self.config.get('scenes', [])
            themes = self.config.get('themes', {})

        print("=" * 70)
        print("📋 VKTRS Storyboard")
        print("=" * 70)
        print(f"Project: {params.get('project_name', 'Unnamed')}")
        print(f"Audio: {params.get('audio_fpath', 'Not set')}")
        duration = params.get('audio_duration') or 0
        print(f"Duration: {float(duration):.1f}s")
        print(f"Scenes: {len(scenes)}")
        print(f"Themes: {len(themes)}")
        print(f"FPS: {params.get('fps', 12)}")
        print(f"Resolution: {params.get('resolution', [512, 512])}")

        if detailed and scenes:
            print("\n" + "=" * 70)
            print("Scenes:")
            print("=" * 70)
            for i, scene in enumerate(scenes):
                duration = scene.get('end_time', 0) - scene.get('start_time', 0)
                print(f"\n{i}: {scene.get('start_time', 0):.2f}s - {scene.get('end_time', 0):.2f}s ({duration:.2f}s)")

                text_val = scene.get('text', '')
                if text_val:
                    text = text_val[:60] + "..." if len(text_val) > 60 else text_val
                    print(f"   Text: {text}")

                prompt_val = scene.get('prompt', '')
                if prompt_val:
                    prompt = prompt_val[:60] + "..." if len(prompt_val) > 60 else prompt_val
                    print(f"   Prompt: {prompt}")

                if 'theme' in scene:
                    print(f"   Theme: {scene['theme']}")

        print("=" * 70)

    def add_scene(self, start_time: float, end_time: float, **kwargs):
        """
        Add a scene to the storyboard.

        Args:
            start_time: Scene start time in seconds
            end_time: Scene end time in seconds
            **kwargs: Additional scene properties (prompt, text, theme, etc.)
        """
        scene = {
            'start_time': start_time,
            'end_time': end_time,
            **kwargs
        }

        if OMEGACONF_AVAILABLE:
            self.config.scenes.append(scene)
        else:
            self.config['scenes'].append(scene)

    def add_theme(self, theme_id: str, **properties):
        """
        Add a theme to the storyboard.

        Args:
            theme_id: Unique theme identifier
            **properties: Theme properties (prompt_prefix, style, etc.)
        """
        if OMEGACONF_AVAILABLE:
            self.config.themes[theme_id] = properties
        else:
            self.config['themes'][theme_id] = properties

    def get_scene_count(self) -> int:
        """Get number of scenes in storyboard."""
        if OMEGACONF_AVAILABLE:
            return len(self.config.scenes)
        else:
            return len(self.config.get('scenes', []))

    def get_total_duration(self) -> float:
        """Get total duration covered by scenes."""
        if OMEGACONF_AVAILABLE:
            scenes = self.config.scenes
        else:
            scenes = self.config.get('scenes', [])

        if not scenes:
            return 0.0

        return scenes[-1].get('end_time', 0.0)


# ============================================================================
# Factory Functions
# ============================================================================

def load_storyboard(path: Path) -> Storyboard:
    """
    Load storyboard from YAML file.

    Args:
        path: Path to storyboard.yaml

    Returns:
        Storyboard object
    """
    return Storyboard(config_path=path)


def create_storyboard(project_name: str,
                     audio_fpath: str,
                     **kwargs) -> Storyboard:
    """
    Create new storyboard with basic configuration.

    Args:
        project_name: Project name
        audio_fpath: Path to audio file
        **kwargs: Additional params (fps, resolution, etc.)

    Returns:
        Storyboard object
    """
    sb = Storyboard()

    sb.config['params']['project_name'] = project_name
    sb.config['params']['audio_fpath'] = audio_fpath

    for key, value in kwargs.items():
        sb.config['params'][key] = value

    return sb


def export_to_deforum(storyboard: Storyboard, output_path: Path):
    """
    Export storyboard to Deforum-compatible format.

    Args:
        storyboard: Storyboard to export
        output_path: Output path for deforum settings
    """
    compiled = storyboard.compile()

    # Convert to Deforum format - safely access resolution
    resolution = compiled.get('params', {}).get('resolution')
    if not resolution or len(resolution) < 2:
        # Use default resolution if missing
        resolution = [512, 512]
    
    deforum_settings = {
        'W': resolution[0],
        'H': resolution[1],
        'fps': compiled.get('params', {}).get('fps', 12),
        # Add more Deforum-specific mappings here
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(deforum_settings, f, default_flow_style=False)

    logger.info(f"💾 Exported to Deforum format: {output_path}")
