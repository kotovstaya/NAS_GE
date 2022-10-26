import typing as tp
from dataclasses import dataclass
import numpy as np
import re
from nasge import utils as nasge_uils

from torch import nn


class BaseGrammar:
    def __init__(self, filepath: tp.Optional[str] = None):
        if filepath is not None:
            self.grammar_info = self._load_from_file(filepath)
        self.rules = self.grammar_info["grammar"]
        self.number_mapping = self._get_number_mapping(
            self.grammar_info["number_mapping"])
        self.enter_expr = self.grammar_info["enter_expr"]

    def generate_number(self, number_key):
        return self.number_mapping[number_key](0)

    def get(self, key):
        return self.rules[key]

    @staticmethod
    def _load_from_file(path: str):
        return nasge_uils.load_yaml(path)

    @staticmethod
    def _get_number_mapping(number_mapping_info: tp.Dict[str, tp.Any]) -> tp.Dict[str, callable]:
        mapping = {}
        for name, rng in number_mapping_info.items():
            rng = list(map(int, rng))
            if name.startswith("int"):
                def _f(rng):
                    return lambda x: np.random.choice(range(rng[0], rng[1]), size=1)[0]
            elif name.startswith("float"):
                def _f(rng):
                    return lambda x: np.round(np.random.uniform(rng[0], rng[1], size=1)[0],3)
            mapping[name] = _f(rng)
        return mapping


class ContextFreeGrammar(BaseGrammar):
    def __init__(self, filepath: tp.Optional[str] = None):
        super().__init__(filepath)


class InstanceGenotype:
    def __init__(self,
                 genoelement_range: tp.Optional[tp.Tuple[int, int]] = None,
                 genotype_size: tp.Optional[int] = None,
                 genotype_as_int_seq: tp.Optional[tp.List[int]] = None):
        self.genoelement_range = genoelement_range
        self.genotype_size = genotype_size
        if genotype_as_int_seq is None:
            self.values = self.create()
        else:
            self.values = self.create_from_seq(genotype_as_int_seq)

    def create(self):
        return [
            np.random.choice(
                range(self.genoelement_range[0], self.genoelement_range[1])
            )
            for _ in range(self.genotype_size)
        ]

    def create_from_seq(self, values: tp.List[int]):
        return values


class InstancePhenotype:
    def __init__(self,
                 grammar: BaseGrammar,
                 genotype: InstanceGenotype,
                 default_filler: str = "_|"):
        self.grammar = grammar
        self.genotype = genotype
        self.re_pattern = re.compile(r"<[\w|\_|\-|\d]+>")
        self.default_filler = default_filler

        self.values = self.get_phenotype()

    def get_phenotype(self):
        phenotype = self.grammar.enter_expr
        for geno in self.genotype.values:
            s, e = self._find_place(phenotype)
            substitute = self._get_substitution(phenotype, s, e, geno)
            phenotype = self._replace_element(phenotype, s, e, substitute)
        phenotype = self._fill_incomplete(phenotype)
        phenotype = self._count_layers(phenotype)
        phenotype = phenotype.split("$")
        phenotype = self._drop_invalid_layers(phenotype)
        phenotype = self._fill_numbers(phenotype)
        phenotype = list(map(self._create_dict_with_parameters, phenotype))
        return phenotype

    def _create_dict_with_parameters(self, elem):
        layer_num, layer_name, *tail = elem.split(",")
        tail = ",".join(tail)
        if tail == "empty":
            tail = "{}"
        else:
            new_parameter_string = []
            for v in tail.split(","):
                new_parameter_string.append(f"'{v.split(':')[0]}': {v.split(':')[1]}")
            tail = "{" + ','.join(new_parameter_string) + "}"
        return ",".join([layer_num, layer_name]), eval(tail)

    def _get_substitution(self,
                          phenotype: str,
                          s: tp.Optional[int],
                          e: tp.Optional[int],
                          geno: int) -> tp.Optional[str]:
        if s is not None:
            substitutes = self.grammar.get(phenotype[s:e])
            return substitutes[geno % len(substitutes)]
        return None

    def _find_place(self,
                    phenotype: str,
                    re_pattern: re.Pattern = None
                    ) -> tp.Tuple[tp.Optional[int], tp.Optional[int]]:
        if re_pattern is None:
            re_pattern = self.re_pattern
        p = re.search(re_pattern, phenotype)
        return (p.start(), p.end()) if p is not None else (None, None)

    @staticmethod
    def _replace_element(phenotype, s, e, replacer) -> str:
        if s is not None:
            if callable(replacer):
                replacer = replacer(0)
            phenotype = phenotype[:s] + str(replacer) + phenotype[e:]
        return phenotype

    def _fill_incomplete(self, phenotype: str):
        s = True
        while s is not None:
            s, e = self._find_place(phenotype)
            phenotype = self._replace_element(phenotype, s, e,
                                              self.default_filler)
        return phenotype

    def _count_layers(self, phenotype: str) -> str:
        s = True
        count = 0
        while s is not None:
            s, e = self._find_place(phenotype, re.compile(r"layer"))
            phenotype = self._replace_element(phenotype, s, e,
                                              f"{count}")
            count += 1
        return phenotype

    def _get_new_layer(self, layer_num, layer_type, tail) -> str:
        return ",".join([layer_num, layer_type] + tail)

    def _update_fc(self,
                   layer_num,
                   layer_type,
                   tail,
                   output_size) -> tp.Tuple[str, str]:
        new_tail = []
        if output_size is None:
            input_size = str(self.grammar.generate_number(tail[0].split(":")[1]))
            new_tail.append(f"in_features:{input_size}")
        else:
            new_tail.append(f"in_features:{output_size}")
        output_size = str(self.grammar.generate_number(tail[1].split(":")[1]))
        new_tail.append(f"out_features:{output_size}")
        return output_size, self._get_new_layer(layer_num, layer_type, new_tail)

    def _update_do(self,
                   layer_num,
                   layer_type,
                   tail,
                   output_size) -> tp.Tuple[str, str]:
        new_tail = [f"p:{self.grammar.generate_number(tail[0].split(':')[1])}"]
        return output_size, self._get_new_layer(layer_num, layer_type, new_tail)

    def _update_bn(self,
                   layer_num,
                   layer_type,
                   tail,
                   output_size) -> tp.Tuple[str, str]:
        new_tail = [f"num_features:{output_size}"]
        return output_size, self._get_new_layer(layer_num, layer_type, new_tail)

    def _update_unknown(self,
                        layer_num,
                        layer_type,
                        tail,
                        output_size) -> tp.Tuple[int, str]:
        return output_size, self._get_new_layer(layer_num, layer_type, tail)

    def _update_layer(self,
                      layer_num,
                      layer_type,
                      tail,
                      output_size) -> tp.Tuple[int, str]:
        if layer_type == 'fc':
            output_size, new_layer = self._update_fc(layer_num, layer_type,
                                                     tail, output_size)
        elif layer_type == "do":
            output_size, new_layer = self._update_do(layer_num, layer_type,
                                                     tail, output_size)
        elif layer_type == "bn":
            output_size, new_layer = self._update_bn(layer_num, layer_type,
                                                     tail, output_size)
        else:
            output_size, new_layer = self._update_unknown(layer_num,
                                                          layer_type, tail,
                                                          output_size)
        return output_size, new_layer

    def _fill_numbers(self, layers: tp.List[str]) -> tp.List[str]:
        result = []
        output_size = None
        for ix, layer_info in enumerate(layers):
            layer_num, layer_type, *tail = layer_info.split(",")
            output_size, new_layer = self._update_layer(layer_num,
                                                        layer_type,
                                                        tail,
                                                        output_size)
            result.append(new_layer)

        return result

    def _drop_invalid_layers(self, phenotype: tp.List[str]) -> tp.List[str]:
        """

        :param phenotype:
        :return:
        """
        return [el for el in phenotype if "_|" not in el and el != ""]


class InstancePyTorchModel:
    def __init__(self, phenotype: InstancePhenotype, **kwargs):
        self.repr_2_layer_mapping = {
            "fc": nn.Linear,
            "sigmoid": nn.Sigmoid,
            "bn": nn.BatchNorm1d,
            "do": nn.Dropout1d,
            "tanh": nn.Tanh,
            "relu": nn.ReLU
        }

        (self.model,
         self.input_size,
         self.output_size) = self._parse_phenotype(phenotype)

    def _parse_phenotype(self, phenotype: InstancePhenotype
                         ) -> tp.Tuple[nn.Sequential, int, int]:
        modules = []
        input_size = 0
        output_size = 0
        for layer_info, layer_params in phenotype.values:
            layer_num, layer_name = layer_info.split(",")
            if layer_num == '0' and layer_name == 'fc':
                input_size = layer_params["in_features"]
            if layer_name == 'fc':
                output_size = layer_params["out_features"]
            layer = self.repr_2_layer_mapping[layer_name](**layer_params)
            modules.append(layer)
        return nn.Sequential(*modules), input_size, output_size


@dataclass()
class Individual:
    grammar: ContextFreeGrammar
    genotype: InstanceGenotype
    phenotype: InstancePhenotype
    model: InstancePyTorchModel


class InstanceBuilder:
    def __init__(
            self,
            grammar_builder: tp.Type[ContextFreeGrammar],
            genotype_builder: tp.Type[InstanceGenotype],
            phenotype_builder: tp.Type[InstancePhenotype],
            model_builder: tp.Type[InstancePyTorchModel],
            grammar_params: tp.Dict[tp.Any, tp.Any],
            genotype_params: tp.Dict[tp.Any, tp.Any],
            phenotype_params: tp.Dict[tp.Any, tp.Any],
            model_params: tp.Dict[tp.Any, tp.Any]):
        self.grammar = grammar_builder(**grammar_params)
        self.genotype_builder = genotype_builder
        self.phenotype_builder = phenotype_builder
        self.model_builder = model_builder

        self.genotype_params = genotype_params
        self.phenotype_params = phenotype_params
        self.model_params = model_params

    def _build_only_phenotype_and_model(self, genotype):
        phenotype = self.phenotype_builder(grammar=self.grammar,
                                           genotype=genotype,
                                           **self.phenotype_params)
        model = self.model_builder(phenotype=phenotype, **self.model_params)
        return phenotype, model

    def build_with_genotype_as_seq(self, genotype_values: tp.List[int]):
        genotype = self.genotype_builder(genotype_as_int_seq=genotype_values)
        return self.build_with_genotype(genotype)

    def build_with_genotype(self, genotype):
        phenotype, model = self._build_only_phenotype_and_model(genotype)
        return Individual(self.grammar, genotype, phenotype, model)

    def build(self):
        genotype = self.genotype_builder(**self.genotype_params)
        return self.build_with_genotype(genotype)
