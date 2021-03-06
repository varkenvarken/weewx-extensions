"""
To use:
    1. Stop weewxd
    2. Put this file in your user subdirectory.
    3. In weewx.conf, subsection [Engine][[Services]], add SunMoonService to the list
    "xtype_services". For example, this means changing this

        [Engine]
            [[Services]]
                xtype_services = weewx.wxxtypes.StdWXXTypes, weewx.wxxtypes.StdPressureCooker, weewx.wxxtypes.StdRainRater

    to this:

        [Engine]
            [[Services]]
                xtype_services = weewx.wxxtypes.StdWXXTypes, weewx.wxxtypes.StdPressureCooker, weewx.wxxtypes.StdRainRater, user.sunmoon.SunMoonService

    4. Restart weewxd

"""
import math

import weewx
import weewx.units
import weewx.xtypes
from weewx.engine import StdService
from weewx.units import ValueTuple
from weewx.almanac import Almanac

class SunMoon(weewx.xtypes.XType):

    def __init__(self, alt, lat, lng):
        self.alt = alt
        self.lat = lat
        self.lng = lng
    
    def get_scalar(self, obs_type, record, db_manager):
        # We only know how to calculate some sun and moon related stuff. For everything else, raise an exception UnknownType
        if obs_type not in ('sun_altitude', 'sun_azimuth' ):
            raise weewx.UnknownType(obs_type)

        # we only report sun values if the sun is above the horizon
        almanac = Almanac(record['dateTime'], self.lat, self.lng, 0)  # TODO convert alt  to meters
        sun_altitude = almanac.sun.alt
        if obs_type == 'sun_altitude':
            value = ValueTuple(sun_altitude if sun_altitude >= 0 else None , 'degree_compass', 'group_direction')
        elif obs_type == 'sun_azimuth':
            value = ValueTuple(almanac.sun.az if sun_altitude >= 0 else None, 'degree_compass', 'group_direction')

        # We have the calculated values as ValueTuples. Convert them back to the units used by
        # the incoming record and return it
        return weewx.units.convertStd(value, record['usUnits'])


class SunMoonService(StdService):
    """ WeeWX service whose job is to register the XTypes extension SunMoon with the
    XType system.
    """

    def __init__(self, engine, config_dict):
        super(SunMoonService, self).__init__(engine, config_dict)

        altitude_vt = engine.stn_info.altitude_vt
        latitude_f = engine.stn_info.latitude_f
        longitude_f = engine.stn_info.longitude_f

        # Instantiate an instance of SunMoon:
        self.sm = SunMoon(altitude_vt, latitude_f, longitude_f)
        # Register it:
        weewx.xtypes.xtypes.append(self.sm)

    def shutDown(self):
        # Remove the registered instance:
        weewx.xtypes.xtypes.remove(self.sm)


# Tell the unit system what group our new observation types belong to:
weewx.units.obs_group_dict['sun_altitude'] = "group_direction"
weewx.units.obs_group_dict['sun_azimuth'] = "group_direction"

