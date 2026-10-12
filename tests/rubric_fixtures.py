"""Generic synthetic test rubrics; these are never application defaults."""

UNCERTAINTY_RUBRIC = {
    "title": "Synthetic completeness and readability",
    "instructions": "Assess the explanation of a simple process.",
    "criteria": [
        {"id": "completeness", "name": "Completeness", "description": "Assess the process steps explained.",
         "levels": [
             {"id": "advanced", "label": "Complete", "description": "Explains all steps and connections.", "points": 8},
             {"id": "adequate", "label": "Adequate", "description": "Explains the essential steps.", "points": 5},
             {"id": "limited", "label": "Limited", "description": "Explains one relevant step.", "points": 2},
             {"id": "missing", "label": "Missing", "description": "Explains no relevant step.", "points": 0},
         ]},
        {"id": "readability", "name": "Readability", "description": "Assess whether the explanation can be understood.",
         "levels": [
             {"id": "clear", "label": "Readable", "description": "The process is understandable.", "points": 2},
             {"id": "unclear", "label": "Unclear", "description": "The process cannot be understood.", "points": 0},
         ]},
    ],
}

GENERIC_RUBRIC = {'title': 'Short response · evidence and reasoning',
 'instructions': 'Explain why urban trees help cities manage heat. Support your explanation with '
                 'specific mechanisms or examples.',
 'criteria': [{'id': 'understanding',
               'name': 'Understanding',
               'description': 'Assess the accuracy of the explanation of how trees reduce urban heat.',
               'levels': [{'id': 'strong',
                           'label': 'Strong',
                           'description': 'Accurately explains both shade and evaporative cooling.',
                           'points': 4},
                          {'id': 'developing',
                           'label': 'Developing',
                           'description': 'Accurately explains one cooling mechanism, with limited '
                                          'detail.',
                           'points': 2},
                          {'id': 'missing',
                           'label': 'Not demonstrated',
                           'description': 'Gives no accurate explanation of a cooling mechanism.',
                           'points': 0}]},
              {'id': 'evidence',
               'name': 'Support and reasoning',
               'description': 'Assess whether examples are connected to the claim about urban heat.',
               'levels': [{'id': 'strong',
                           'label': 'Strong',
                           'description': 'Connects a specific, relevant example to how cooling occurs.',
                           'points': 4},
                          {'id': 'developing',
                           'label': 'Developing',
                           'description': 'Includes a relevant example but only weakly connects it to '
                                          'cooling.',
                           'points': 2},
                          {'id': 'missing',
                           'label': 'Not demonstrated',
                           'description': 'Includes no relevant example or supporting reasoning.',
                           'points': 0}]},
              {'id': 'clarity',
               'name': 'Clarity',
               'description': 'Assess whether the response communicates its ideas clearly.',
               'levels': [{'id': 'strong',
                           'label': 'Clear',
                           'description': 'Communicates a coherent explanation in understandable '
                                          'sentences.',
                           'points': 2},
                          {'id': 'developing',
                           'label': 'Partly clear',
                           'description': 'Meaning is understandable but organization or wording is '
                                          'unclear in places.',
                           'points': 1},
                          {'id': 'missing',
                           'label': 'Unclear',
                           'description': 'Meaning cannot be understood or the response is unrelated.',
                           'points': 0}]}]}
