#Objective, The Main objective of this python file is to create the ENCAPSULATION LAYER (Irrespective of each client)

#Import the required libraries
import pandas as pd
import numpy as np
from collections import OrderedDict


map_ampm_fields = [
    ('Opened_at'     , 'Open Time (Date Only)'),
    ('Closed_at'     , 'Close Time (Date Only)'),
    ('Resolution_Time', 'Resolution Time'),
    ('Priority'      , 'Priority'),
    ('Department'    , 'Assignment Group'),
    ('Customer' 		, 'Client Ticket')
]

map_ampm_fields = OrderedDict(map_ampm_fields)


map_servicenow_fields = [
    ('Opened_at'     , 'opened_at'),
    ('Closed_at'     , 'closed_at'),
    ('Priority'      , 'priority'),
    ('Department'    , 'assignment_group'),
    ('Customer' 		, 'Client Ticket')
]
map_servicenow_fields = OrderedDict(map_servicenow_fields)


def semantic_columns(APP_TYPE):
	if APP_TYPE.lower() == 'ampm':
		return list(map_ampm_fields.values()), list(map_ampm_fields.keys())
	if APP_TYPE.lower() == 'servicenow':
		return list(map_servicenow_fields.values()), list(map_servicenow_fields.keys())