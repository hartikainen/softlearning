from lxml import etree
from dm_control.suite import common
from dm_control.utils import xml_tools
from dm_control.suite.point_mass import SUITE


_WALLS = ('wall_x', 'wall_y', 'wall_neg_x', 'wall_neg_y')
KEY_GEOM_NAMES = [
    'pointmass',
]


def make_model(walls_and_target=False, actuator_type='motor'):
    xml_string = common.read_model('point_mass.xml')
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)
    # Remove walls, ball and target.
    if not walls_and_target:
        for wall in _WALLS:
            wall_geom = xml_tools.find_element(mjcf, 'geom', wall)
            wall_geom.getparent().remove(wall_geom)

        # Remove target.
        target_site = xml_tools.find_element(mjcf, 'geom', 'target')
        target_site.getparent().remove(target_site)

    # Rename ground geom to floor
    mjcf.find(".//geom[@name='ground']").attrib['name'] = 'floor'

    # Unlimit the joints
    default_joint_element = mjcf.find('.//default').find(".//joint")
    default_joint_element.attrib['limited'] = 'false'
    default_joint_element.attrib.pop('range')

    # <site name='torso_site' pos='0 0 2' size='5' rgba='1 1 1 1' group='4'/>
    pointmass_site = etree.Element(
        'site',
        name='pointmass_site',
        pos='0, 0, 0',
        size='1',
        rgba='1 1 1 1',
        group='4',
    )

    pointmass_body = mjcf.find(".//body[@name='pointmass']")
    pointmass_body.insert(0, pointmass_site)

    sensor = etree.Element('sensor')
    velocimeter_sensor = etree.Element(
        'velocimeter',
        name='sensor_torso_vel',
        site='pointmass_site'
    )

    sensor.insert(0, velocimeter_sensor)
    mjcf.insert(-1, sensor)

    if actuator_type in ('motor', 'velocity'):
        actuator_node = mjcf.find('actuator')
        t1_motor = actuator_node.find(".//motor[@name='t1']")
        t1_motor.tag = actuator_type
        t1_motor.attrib['ctrlrange'] = '-1 1'
        t1_motor.attrib['forcerange'] = '-1000000000 1000000000'
        t1_motor.attrib['gear'] = '1.0'
        t2_motor = actuator_node.find(".//motor[@name='t2']")
        t2_motor.tag = actuator_type
        t2_motor.attrib['ctrlrange'] = '-1 1'
        t2_motor.attrib['forcerange'] = '-1000000000 1000000000'
        t2_motor.attrib['gear'] = '1.0'

    return etree.tostring(mjcf, pretty_print=True)
