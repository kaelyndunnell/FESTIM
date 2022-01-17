from fenics import *
import csv
import sys
import os
import sympy as sp
import json
import numpy as np


class Export:
    def __init__(self) -> None:
        pass


class Exports:
    def __init__(self, exports=[]) -> None:
        self.exports = exports


def export_txt(filename, function, functionspace):
    '''
    Exports a 1D function into a txt file.
    Arguments:
    - filemame : str
    - function : fenics.Function()
    - functionspace: fenics.FunctionSpace()
    Returns:
    - True on sucess
    '''
    export = project(function, functionspace)
    busy = True
    x = interpolate(Expression('x[0]', degree=1), functionspace)
    while busy is True:
        try:
            np.savetxt(filename + '.txt', np.transpose(
                        [x.vector()[:], export.vector()[:]]))
            return True
        except OSError as err:
            print("OS error: {0}".format(err))
            print("The file " + filename + ".txt might currently be busy."
                  "Please close the application then press any key.")
            input()


def export_profiles(res, exports, t, dt, functionspace):
    '''
    Exports 1D profiles in txt files.
    Arguments:
    - res: list, contains fenics.Functions
    - exports: dict, contains parameters
    - t: float, time
    - dt: fenics.Constant(), stepsize
    - functionspace: fenics.FunctionSpace()
    Returns:
    - dt: fenics.Constant(), stepsize
    '''
    functions = exports['txt']['functions']
    labels = exports['txt']['labels']
    if len(functions) != len(labels):
        raise NameError("Number of functions to be exported "
                        "doesn't match number of labels in txt exports")
    if len(functions) > len(res):
        raise NameError("Too many functions to export "
                        "in txt exports")
    solution_dict = {
        'solute': res[0],
        'retention': res[len(res)-2],
        'T': res[len(res)-1],
    }
    times = sorted(exports['txt']['times'])
    end = True
    for time in times:
        if t == time:
            if times.index(time) != len(times)-1:
                next_time = times[times.index(time)+1]
                end = False
            else:
                end = True
            for i in range(len(functions)):
                if functions[i].isdigit() is True:
                    solution = res[int(functions[i])]
                elif functions[i] in solution_dict:
                    solution = solution_dict[functions[i]]
                else:
                    raise ValueError(
                        "function " + functions[i] + " is unknown")
                label = labels[i]
                export_txt(
                    exports["txt"]["folder"] + '/' + label + '_' +
                    str(t) + 's',
                    solution, functionspace)
            break
        if t < time:
            next_time = time
            end = False
            break
    if end is False:
        if t + float(dt) > next_time:
            dt.assign(time - t)
    return dt


def define_xdmf_files(exports):
    '''
    Returns a list of XDMFFile
    Arguments:
    - exports: dict, contains parameters
    Returns:
    - files: list, contains the fenics.XDMFFile() objects
    '''
    if len(exports['xdmf']['functions']) != len(exports['xdmf']['labels']):
        raise NameError("Number of functions to be exported "
                        "doesn't match number of labels in xdmf exports")
    if exports["xdmf"]["folder"] == "":
        raise ValueError("folder value cannot be an empty string")
    if type(exports["xdmf"]["folder"]) is not str:
        raise TypeError("folder value must be of type str")
    files = list()
    for i in range(0, len(exports["xdmf"]["functions"])):
        u_file = XDMFFile(exports["xdmf"]["folder"]+'/' +
                          exports["xdmf"]["labels"][i] + '.xdmf')
        u_file.parameters["flush_output"] = True
        u_file.parameters["rewrite_function_mesh"] = False
        files.append(u_file)
    return files


def treat_value(d):
    '''
    Recursively converts as string the sympy objects in d
    Arguments: d, dict
    Returns: d, dict
    '''

    T = sp.symbols('T')
    if type(d) is dict:
        d2 = {}
        for key, value in d.items():
            if isinstance(value, tuple(sp.core.all_classes)):
                value = str(sp.printing.ccode(value))
                d2[key] = value
            elif callable(value):  # if value is fun
                d2[key] = str(sp.printing.ccode(value(T)))
            elif type(value) is dict or type(value) is list:
                d2[key] = treat_value(value)
            else:
                d2[key] = value
    elif type(d) is list:
        d2 = []
        for e in d:
            e2 = treat_value(e)
            d2.append(e2)
    else:
        d2 = d

    return d2


def export_parameters(parameters):
    '''
    Dumps parameters dict in a json file.
    '''
    json_file = parameters["exports"]["parameters"]
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    if json_file.endswith(".json") is False:
        json_file += ".json"
    param = treat_value(parameters)
    with open(json_file, 'w') as fp:
        json.dump(param, fp, indent=4, sort_keys=True)
    return True
