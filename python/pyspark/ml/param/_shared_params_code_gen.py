#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

header = """#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#"""

# Code generator for shared params (shared.py). Run under this folder with:
# python _shared_params_code_gen.py > shared.py


def _gen_param_code(name, doc, defaultValueStr):
    """
    Generates Python code for a shared param class.

    :param name: param name
    :param doc: param doc
    :param defaultValueStr: string representation of the default value
    :return: code string
    """
    # TODO: How to correctly inherit instance attributes?
    template = '''class Has$Name(Params):
    """
    Mixin for param $name: $doc.
    """

    # a placeholder to make it appear in the generated doc
    $name = Param(Params._dummy(), "$name", "$doc")

    def __init__(self):
        super(Has$Name, self).__init__()
        #: param for $doc
        self.$name = Param(self, "$name", "$doc")
        if $defaultValueStr is not None:
            self._setDefault($name=$defaultValueStr)

    def set$Name(self, value):
        """
        Sets the value of :py:attr:`$name`.
        """
        self.paramMap[self.$name] = value
        return self

    def get$Name(self):
        """
        Gets the value of $name or its default value.
        """
        return self.getOrDefault(self.$name)'''

    Name = name[0].upper() + name[1:]
    return template \
        .replace("$name", name) \
        .replace("$Name", Name) \
        .replace("$doc", doc) \
        .replace("$defaultValueStr", str(defaultValueStr))

if __name__ == "__main__":
    print(header)
    print("\n# DO NOT MODIFY THIS FILE! It was generated by _shared_params_code_gen.py.\n")
    print("from pyspark.ml.param import Param, Params\n\n")
    shared = [
        ("maxIter", "max number of iterations", None),
        ("regParam", "regularization constant", None),
        ("featuresCol", "features column name", "'features'"),
        ("labelCol", "label column name", "'label'"),
        ("predictionCol", "prediction column name", "'prediction'"),
        ("rawPredictionCol", "raw prediction column name", "'rawPrediction'"),
        ("inputCol", "input column name", None),
        ("inputCols", "input column names", None),
        ("outputCol", "output column name", None),
        ("numFeatures", "number of features", None),
        ("elasticNetParam", "the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, " +
         "the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.", None)]
    code = []
    for name, doc, defaultValueStr in shared:
        code.append(_gen_param_code(name, doc, defaultValueStr))
    print("\n\n\n".join(code))
