from configparser import ConfigParser
from pathlib import Path

parser = ConfigParser()

parser.read('configurations.ini')

if parser.getint('project', 'verbose'):
    print('io:')
for name in parser.options('io'):
    parser['io'][name] = parser.get('io', name)

    if parser.getint('project', 'verbose'):
        print('  {:<12} : -> {}'.format(
            name, parser['io'][name]))

print('Integers:')
for name in parser.options('io'):
    string_value = parser.get('ints', name)
    value = parser.getint('ints', name)
    print('  {:<12} : {!r:<7} -> {}'.format(
        name, string_value, value))

print('\nFloats:')
for name in parser.options('floats'):
    string_value = parser.get('floats', name)
    value = parser.getfloat('floats', name)
    print('  {:<12} : {!r:<7} -> {:0.2f}'.format(
        name, string_value, value))

print('\nBooleans:')
for name in parser.options('booleans'):
    string_value = parser.get('booleans', name)
    value = parser.getboolean('booleans', name)
    print('  {:<12} : {!r:<7} -> {}'.format(
        name, string_value, value))
