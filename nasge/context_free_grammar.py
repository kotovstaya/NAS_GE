import typing as tp
from dataclasses import dataclass
import numpy as np
import re

from torch import nn
from nasge.genetic_algorithm import GAEvolution, Individual


@dataclass
class DefaultGrammar:
    """

    """
    grammar: tp.Optional[tp.Dict[str, tp.List[str]]] = None

    def init_grammar(self):
        self.grammar = {
            "<expr>": [
                "<fc>$",
                "<fc>$<expr>$"
                "<fc>$<bn>$",
                "<fc>$<do>$<bn>$",
                "<fc>$<bn>$<do>$",
                "<fc>$<bn>$<expr>$",
                "<fc>$<bn>$<do>$<expr>$",
                "<fc>$<act>$<bn>$",
                "<fc>$<act>$<bn>$<do>$",
                "<fc>$<act>$<bn>$<expr>$",
                "<fc>$<act>$<bn>$<do>$<expr>$",
                "<fc>$<bn>$<act>$",
                "<fc>$<bn>$<act>$<do>$",
                "<fc>$<bn>$<act>$<expr>$",
                "<fc>$<bn>$<act>$<do>$<expr>$",
                "<fc>$<bn>$<do>$<act>$",
                "<fc>$<bn>$<do>$<act>$<expr>$",
            ],
            "<fc>": ["layer,fc,in_features:<in_ftrs>,out_features:<out_ftrs>"],
            "<bn>": ["layer,bn,num_features:<num_ftrs>"],
            "<do>": ["layer,do,p:float_rnd_0-1"],
            "<in_ftrs>": ["int_rnd_10-20", "int_rnd_20-30", "int_rnd_30-40"],
            "<act>": ["layer,sigmoid,empty", "layer,tanh,empty", "layer,relu,empty"],
            "<num_ftrs>": ["int_rnd_10-20", "int_rnd_20-30"],
            "<out_ftrs>": ["int_rnd_10-20", "int_rnd_20-30", "int_rnd_30-40"],
        }

    def get(self, key: str) -> tp.List[str]:
        return self.grammar[key]

    def number_mapping(self, number_key: str):
        mapping = {
            "int_rnd_10-20": lambda x: np.random.choice(range(10, 20), size=1)[0],
            "int_rnd_20-30": lambda x: np.random.choice(range(20, 30), size=1)[0],
            "int_rnd_30-40": lambda x: np.random.choice(range(30, 40), size=1)[0],
            "float_rnd_0-1": lambda x: np.round(np.random.random(), 3)
        }
        return mapping[number_key](0)


class ContextFreeParser:
    """
    В момент инициализации принимает грамматику
    internal_representation - умеет из грамматики и генотипа получить внутреннее представление фенотипа

    """
    def __init__(self,
                 grammar: DefaultGrammar,
                 input_size: int,
                 output_size: int,
                 default_filler: str = "_|",
                 enter_expr: str = "<expr>",
                 ):
        self.grammar = grammar
        self.input_size = input_size
        self.output_size = output_size
        self.re_pattern = re.compile(r"<[\w|\_|\-|\d]+>")
        self.default_filler = default_filler
        self.enter_expr = enter_expr

    def _get_substitution(self,
                          phenotype: str,
                          s: tp.Optional[int],
                          e: tp.Optional[int],
                          geno: int) -> tp.Optional[str]:
        """

        :param phenotype:
        :param s:
        :param e:
        :param geno:
        :return:
        """
        if s is not None:
            substitutes = self.grammar.get(phenotype[s:e])
            return substitutes[geno % len(substitutes)]
        return None

    def _find_place(self,
                    phenotype: str,
                    re_pattern: re.Pattern = None) -> tp.Tuple[
                                                            tp.Optional[int],
                                                            tp.Optional[int]]:
        """

        :param phenotype:
        :param re_pattern:
        :return:
        """
        if re_pattern is None:
            re_pattern = self.re_pattern
        p = re.search(re_pattern, phenotype)
        return (p.start(), p.end()) if p is not None else (None, None)

    @staticmethod
    def _replace_element(phenotype, s, e, replacer) -> str:
        """

        :param phenotype:
        :param s:
        :param e:
        :param replacer:
        :return:
        """
        if s is not None:
            if callable(replacer):
                replacer = replacer(0)
            phenotype = phenotype[:s] + str(replacer) + phenotype[e:]
        return phenotype

    def _fill_incomplete(self, phenotype: str):
        """

        :param phenotype:
        :return:
        """
        s = True
        while s is not None:
            s, e = self._find_place(phenotype)
            phenotype = self._replace_element(phenotype, s, e,
                                              self.default_filler)
        return phenotype

    def _count_layers(self, phenotype: str) -> str:
        """

        :param phenotype:
        :return:
        """
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

    def _update_fc(self, layer_num, layer_type, tail, output_size) -> tp.Tuple[str, str]:
        """

        :param layer_num:
        :param layer_type:
        :param tail:
        :param output_size:
        :return:
        """
        new_tail = []
        if layer_num == '0':
            new_tail.append(f"in_features:{self.input_size}")
        else:
            new_tail.append(f"in_features:{output_size}")
        output_size = str(self.grammar.number_mapping(tail[1].split(":")[1]))
        new_tail.append(f"out_features:{output_size}")
        return output_size, self._get_new_layer(layer_num, layer_type, new_tail)

    def _update_do(self, layer_num, layer_type, tail, output_size) -> tp.Tuple[str, str]:
        """

        :param layer_num:
        :param layer_type:
        :param tail:
        :param output_size:
        :return:
        """
        new_tail = [f"p:{self.grammar.number_mapping(tail[0].split(':')[1])}"]
        return output_size, self._get_new_layer(layer_num, layer_type, new_tail)

    def _update_bn(self, layer_num, layer_type, tail, output_size) -> tp.Tuple[str, str]:
        """

        :param layer_num:
        :param layer_type:
        :param tail:
        :param output_size:
        :return:
        """
        new_tail = [f"num_features:{output_size}"]
        return output_size, self._get_new_layer(layer_num, layer_type, new_tail)

    def _update_unknown(self, layer_num, layer_type, tail, output_size) -> tp.Tuple[int, str]:
        """

        :param layer_num:
        :param layer_type:
        :param tail:
        :param output_size:
        :return:
        """
        return output_size, self._get_new_layer(layer_num, layer_type, tail)

    def _update_layer(self, layer_num, layer_type, tail, output_size) -> tp.Tuple[int, str]:
        """

        :param layer_num:
        :param layer_type:
        :param tail:
        :param output_size:
        :return:
        """
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
        """

        :param layers:
        :return:
        """
        result = []
        output_size = None
        for ix, layer_info in enumerate(layers):
            layer_num, layer_type, *tail = layer_info.split(",")
            output_size, new_layer = self._update_layer(layer_num, layer_type, tail, output_size)
            result.append(new_layer)
        output_layers = [
            ",".join(
                map(
                    str,
                    [
                        len(layers),
                        "fc",
                        f"in_features:{output_size}",
                        f"out_features:{self.output_size}"]
                )
            ),
            ",".join(map(str, [len(layers)+1, "sigmoid", "empty"]))
        ]

        result += output_layers
        return result

    def _drop_invalid_layers(self, phenotype):
        """

        :param phenotype:
        :return:
        """
        return [el for el in phenotype if "_|" not in el and el != ""]

    def internal_representation(self, genotype: tp.List[int]) -> tp.List[str]:
        """

        :param genotype:
        :return:
        """
        phenotype = self.enter_expr
        for geno in genotype:
            s, e = self._find_place(phenotype)
            substitute = self._get_substitution(phenotype, s, e, geno)
            phenotype = self._replace_element(phenotype, s, e, substitute)
        phenotype = self._fill_incomplete(phenotype)
        phenotype = self._count_layers(phenotype)
        phenotype = phenotype.split("$")
        phenotype = self._drop_invalid_layers(phenotype)
        phenotype = self._fill_numbers(phenotype)
        return phenotype


class PyTorchModelBuilder:
    def __init__(self):
        self.repr_2_layer_mapping = None
        self._init_mapping()

    def _init_mapping(self):
        repr_2_layer_mapping = {
            "fc": nn.Linear,
            "sigmoid": nn.Sigmoid,
            "bn": nn.BatchNorm1d,
            "do": nn.Dropout1d,
            "tanh": nn.Tanh,
            "relu": nn.ReLU
        }
        self.repr_2_layer_mapping = repr_2_layer_mapping

    @staticmethod
    def _create_dict_with_parameters(tail: str) -> tp.Dict[str, tp.Any]:
        """

        :param tail:
        :return:
        """
        if tail == "empty":
            tail = "{}"
        else:
            new_parameter_string = []
            for v in tail.split(","):
                new_parameter_string.append(f"'{v.split(':')[0]}': {v.split(':')[1]}")
            tail = "{" + ','.join(new_parameter_string) + "}"
        return eval(tail)

    def build_model(self, internal_representation: tp.List[str]) -> nn.Sequential:
        """

        :param internal_representation:
        :return:
        """
        modules = []
        for layer_info in internal_representation:
            layer_num, layer_name, *tail = layer_info.split(",")
            tail = ",".join(tail)
            params = self._create_dict_with_parameters(tail)

            layer = self.repr_2_layer_mapping[layer_name](**params)
            modules.append(layer)
        model = nn.Sequential(*modules)
        return model


class PyTorchModelWrapper:
    def __init__(self,
                 grammer_class,
                 model_builder,
                 parser_class,
                 input_size: int,
                 output_size: int,
                 phenotype_size: int,
                 phenotype_range=(0, 10)):
        self.input_size = input_size
        self.output_size = output_size
        self.phenotype_size = phenotype_size
        self.phenotype_range = phenotype_range

        self.dg = grammer_class()
        self.model_builder = model_builder()
        self.dg.init_grammar()

        self.parser = parser_class(self.dg, input_size, output_size)

    def get_genotype(self) -> tp.List[int]:
        return list(np.random.randint(*self.phenotype_range,
                                      size=self.phenotype_size))

    def get_phenotype(self, genotype) -> tp.List[str]:
        return self.parser.internal_representation(genotype)

    def get_model(self, phenotype) -> nn.Sequential:
        return self.model_builder.build_model(phenotype)

    def get_phenotype_and_model(self, genotype):
        phenotype = self.get_phenotype(genotype)
        model = self.get_model(phenotype)
        return phenotype, model

    def create_instance(self) -> Individual:
        genotype = self.get_genotype()
        phenotype, model = self.get_phenotype_and_model(genotype)
        return Individual(genotype, phenotype, model)


class ContextFreePopulation:
    def __init__(self,
                 grammer_class,
                 model_builder,
                 parser_class,
                 input_size: int,
                 output_size: int,
                 phenotype_size: int,
                 phenotype_range=(0, 10),
                 population_size=10,
                 ):
        self.population_size = population_size
        self.phenotype_range = phenotype_range
        self.ptmw = PyTorchModelWrapper(
            grammer_class=grammer_class,
            model_builder=model_builder,
            parser_class=parser_class,
            input_size=input_size,
            output_size=output_size,
            phenotype_size=phenotype_size,
            phenotype_range=phenotype_range
        )

    def create_population(self) -> tp.List[Individual]:
        return [self.ptmw.create_instance()
                for _ in range(self.population_size)]


if __name__ == "__main__":
    ptmt = ContextFreePopulation(
        grammer_class=DefaultGrammar,
        model_builder=PyTorchModelBuilder,
        parser_class=ContextFreeParser,
        input_size=100,
        output_size=1,
        phenotype_size=30,
        phenotype_range=(0, 30),
        population_size=20
    )

    evolution = GAEvolution(ptmt)
    population = evolution.eval(
        offspring_fraction=0.2,
        crossover_probability=0.5,
        individual_mutate_probability=0.5,
        genoelem_mutate_probability=0.2,
    )

    for pop in population:
        print(pop.genotype)
        # print(pop.phenotype)
        # print(pop.model)
