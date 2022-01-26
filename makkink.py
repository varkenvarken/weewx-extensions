"""
Evapotranspiration Makkink

REQUIRES WeeWX V4.2 OR LATER!

To use:
    1. Stop weewxd
    2. Put this file in your user subdirectory.
    3. In weewx.conf, subsection [Engine][[Services]], add MakkinkService to the list
    "xtype_services". For example, this means changing this

        [Engine]
            [[Services]]
                xtype_services = weewx.wxxtypes.StdWXXTypes, weewx.wxxtypes.StdPressureCooker, weewx.wxxtypes.StdRainRater

    to this:

        [Engine]
            [[Services]]
                xtype_services = weewx.wxxtypes.StdWXXTypes, weewx.wxxtypes.StdPressureCooker, weewx.wxxtypes.StdRainRater, user.makkink.MakkinkService

    4. Restart weewxd

"""
import math

import weewx
import weewx.units
import weewx.xtypes
from weewx.engine import StdService
from weewx.units import ValueTuple


class Makkink(weewx.xtypes.XType):
    
    """ see: https://nl.wikipedia.org/wiki/Referentie-gewasverdamping
    """

    def __init__(self):
        self.λ = 2.45E6  # heat of evaporation in J/Kg of water at 20 degrees C
        self.γ = 0.66    # psychrometer constant mbar/C
        self.C1= 0.65

    def Eref(self, T, Kin):
        """
        Returns evapotranspiration in kg/(m2 * s)
        """
        a = 6.1078
        b = 17.294
        c = 237.73
        s = ((a * b * c) / ((c + T) ** 2)) * math.exp((b * T)/(c+T))  # temperature derivative of saturation vapor pressure
        return (self.C1 * (s / (s + self.γ)) * Kin ) / self.λ
        
    def get_scalar(self, obs_type, record, db_manager):
        # We only know how to calculate 'makkink'. For everything else, raise an exception UnknownType
        if obs_type != 'makkink':
            raise weewx.UnknownType(obs_type)

        # We need outTemp and the radiation in order to do the calculation.
        if 'outTemp' not in record or record['outTemp'] is None:
            raise weewx.CannotCalculate(obs_type)
        if 'radiation' not in record or record['radiation'] is None:
            raise weewx.CannotCalculate(obs_type)

        # We have everything we need. Start by forming a ValueTuple for the outside temperature.
        # To do this, figure out what unit and group the record is in ...
        unit_and_group = weewx.units.getStandardUnitType(record['usUnits'], 'outTemp')
        # ... then form the ValueTuple.
        outTemp_vt = ValueTuple(record['outTemp'], *unit_and_group)

        # same for radiation
        unit_and_group = weewx.units.getStandardUnitType(record['usUnits'], 'radiation')
        # ... then form the ValueTuple.
        radiation_vt = ValueTuple(record['radiation'], *unit_and_group)

        outTemp_C_vt = weewx.units.convert(outTemp_vt, 'degree_C')
        # Get the first element of the ValueTuple. This will be in Celsius:
        T = outTemp_C_vt[0]

        # just to make sure nothing fancy happens we do this for radiation as well
        # even though this is W/m2 in us as well as metric
        radiation_Wm2_vt = weewx.units.convert(radiation_vt, 'watt_per_meter_squared')
        Kin = radiation_Wm2_vt[0]


        makkink = ValueTuple(self.Eref(T, Kin) * 3600, 'mm_per_hour', 'group_rainrate')  # convert from kg/m2*s --> mm/hour
        
        # Convert it back to the units used by
        # the incoming record and return it
        return weewx.units.convertStd(makkink, record['usUnits'])


class MakkinkService(StdService):
    
    def __init__(self, engine, config_dict):
        super(MakkinkService, self).__init__(engine, config_dict)

        # Instantiate an instance of Makkink:
        self.mk = Makkink()
        # Register it:
        weewx.xtypes.xtypes.append(self.mk)

    def shutDown(self):
        # Remove the registered instance:
        weewx.xtypes.xtypes.remove(self.mk)


# Tell the unit system what group our new observation type, 'makkink', belongs to:
weewx.units.obs_group_dict['vapor_p'] = "group_rainrate"
