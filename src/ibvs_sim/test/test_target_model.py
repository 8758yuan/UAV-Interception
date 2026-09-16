"""Physical assumptions for the lightweight balloon target."""

from pathlib import Path
import xml.etree.ElementTree as ET


def _model_root():
    path = Path(__file__).parents[1] / 'models' / 'static_target.sdf'
    return ET.parse(path).getroot().find('model')


def test_balloon_is_dynamic_lightweight_and_floating() -> None:
    model = _model_root()
    assert model.findtext('static') == 'false'
    link = model.find('link')
    assert link.findtext('gravity') == 'false'
    assert float(link.findtext('inertial/mass')) <= 0.05


def test_balloon_contact_is_soft_and_low_friction() -> None:
    collision = _model_root().find('link/collision')
    assert float(collision.findtext('surface/contact/ode/kp')) <= 100.0
    assert float(collision.findtext('surface/friction/ode/mu')) <= 0.05
