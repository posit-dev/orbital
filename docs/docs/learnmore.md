# Learn More

Orbital is made by three primary components, which are usually
performed in sequence:

- The pipeline parser
- The translator
- The optimizer

## The Pipeline Parser

When [orbital.parse_pipeline][] is invoked, the SciKit-Learn pipeline is parsed
and converted to a tree of steps. Each step depends on some input variables,
some constants, emits one or more output variables and has attributes.

### Input Features

As SciKit pipelines don't provide an easy way to inspect all the features that
are required to process the pipeline, the parsing process also requires the
schema of the table the pipeline applies to. You don't have to provide the full
table schema, providing only the columns that constitute the input features on
which the pipeline was originally fitted is sufficient.

The Input features are provided via a `FeatureTypes` dictionary, which is a
normal dictionary mapping column names to [orbital.types][] column types.

```python
{
    "sepal_length": orbital.types.DoubleColumnType(),
    "sepal_width": orbital.types.DoubleColumnType(),
    "petal_length": orbital.types.DoubleColumnType(),
    "petal_width": orbital.types.DoubleColumnType(),
}
```

It has to be noted that as something like `X.Y` is confusing as a reference
in SQL (without futher context is not possible to understand if it refers
to `Y` of table `X` or to a column actually named `X.Y`), thus all column names
are required to avoid dots. This is usually not a problem as columns
in SQL tables are rarely named with dots, but you might have to pay more
attention when fitting the pipeline as the input features must have the same
names of the database columns or SQL translation will fail.

For example for the `sepal` dataset when accessed via SciKit-Learn, 
the fields are all named `sepal.length`, `sepal.width`, and so on... 
this means that before they can be used they must
be renamed to `sepal_length`, `sepal_width` and so on...

### Pipeline Steps

A previously mentioned, a pipeline in Orbital is made of multiple steps
that receive input variables and emit output variables.

More simple pipelines can be seen as a sequence of steps, each step receiving
an input from the previous step and sending an output to the next step.

But more complex pipelines might have steps are a tree, where each steps
has children that are the steps that consume its output variables as their input.

For example a linear regression might look like:

```
    variable1=Sub(
        inputs: merged_columns, Su_Subcst=[5.809166666666666, 3.0616666666666665, 3.7266666666666666, 1.18333333...,
        attributes: 
    )
    multiplied=MatMul(
        inputs: variable1, coef=[-0.11633479416518255, -0.05977785171980231, 0.25491374699772246, 0.5475959...,
        attributes: 
    )
    resh=Add(
        inputs: multiplied, intercept=[0.9916666666666668],
        attributes: 
        )
```

The `Sub` step **receives as input a variable** named `merged_columns`, the variable might come
from a previous step or from the initial set of features.

It also has **a constant input**, the list of values that have to be substracted to each column
in the `merged_columns` variable.

At the end of the step, we will emit a new variable `variable1`, which is the result of
executing that step. In the final resulting SQL you won't find all intermediate output variables.

In most cases those variables are collapsed into SQL expressions and won't be visible in the final SQL,
the only variables you will recognize in the final SQL are the initial inputs and the final outputs.
The SQL might nonetheless create other intermediate columns that it needs to reuse, but those
won't appear in the final results of the query.

Each variable could be a single column or a [orbital.translation.variables.ValueVariablesGroup][],
as a user we will never deal with variable groups, those are exclusively created by intermediate steps
during the translation phase of the Pipeline.

Steps can also have one or more attributes. Those usually configure the step and inform it
of how it should execute when multiple possibilities are available.

For example a `OneHotEncoder` step will have the name of the categories as an attribute

```
    petal_width_cat01out=OneHotEncoder(
      inputs: petal_width_cat01,
      attributes: 
        cats_strings=['narrow', 'wide']
        zeros=1
    )
```

and will emit new variable group where each member of the group is a variable that represents
one of the categories as `1` and `0`.

## Translator

The translator is in charge of accepting a [orbital.ast.ParsedPipeline][] and converting it to
an expression that can then be compiled to one of the supported SQL dialects.

The conversion happens step by step, so each step type has its own translator and generally steps
are not aware of the steps preceding and following them. This allows to keep the implementation
simple and contributing new steps straightfoward. Nonetheless, steps will take care of
"persisting" variables that are commonly reused in future steps. For example a `OneHotEncode`
step will generate an intermediate column for each encoded category, because it can take
for granted that the encoded columns will be used in a future step (otherwise there would
be no reason to encode them at all.)

Typically as users we won't see that intermediate expression, as the [orbital.export_sql][]
function takes care of directly converting it to SQL in the provided dialect:

```python
sql = orbital.export_sql("TABLE_NAME", orbital_pipeline, dialect="duckdb")
```

The conversion of the intermediate expression to SQL happens via [SQLGlot](https://sqlglot.com/sqlglot.html)
and thus the supported dialects are those supported by SQLGlot itself.

At the moment the dialects that are actively tested and verified by the *Continuous Integration*
are `sqlite`, `duckdb` and `postgres` so those should be fairly solid, during the course of
time more dialects will be added. This does not mean that you can't use other dialects,
it simply means that we currently don't test for them and thus we can't guarantee they will work
out of the box.

For a complete list of the available dialects refer to [SQLGlot Dialects](https://sqlglot.com/sqlglot/dialects.html),
to use them with Orbital you can simply provide their name as a lowercase string to the
`export_sql` function.

## Optimizer

Orbital has two different optimizers in place:

- Orbital Pipeline->Expression optimzer
- SQLGlot Expression->SQL optimizer.

### Pipeline to Expression Optimizer

The pipeline to expression optimizer is built into Orbital itself,
and is always active. It takes care of producing expressions that
minimize the amount of operations that a Pipeline requires.

It takes care of folding constants, precomputing operations at compile time,
avoiding `CASE WHEN` statements or `CAST` when possible.

The `Pipeline->Expression` optimizer is invoked directly by the
step translators when they are aware of possible optimizations,
thus it doesn't involve a second pass over the expression.
But this means that its up to developers implementing the steps
to properly invoke the optimizer.

This optimizer cannot be disabled, disabling it can easily lead to 
SQL queries that are too big for your database SQL parser to understand
and execute. Or it can even cause OOMs to the machine compiling the
SQL itself.

For example compiling and running a Gradient Boosted Tree with Orbital
usually takes tens of seconds and takes around 500Mb-1Gb of memory, but
when the `Pipeline->Expression` optimizer is disabled it will take tens of minutes and
will spike to 20-24Gb of consumed ram, thus making the compile process
not viable.

Disabling this optimizer is only meant for debugging reasons, and thus
is not exposed to end users.

### Expression to SQL Optimizer

This is based one the [SQLGlot optimizer](https://sqlglot.com/sqlglot/optimizer/optimizer.html#optimize)
and given an existing expression it takes care of optimizing the SQL itself.

Using it leads to smaller and faster SQL, but it's not perfect and can
sometimes choke on more complex queries. 

If compiling to SQL gets stuck you can try to disable the SQL optimizer
by passing `optimize=False` to [orbital.export_sql][].

We can take as an example the following query resulting from a Linear Regression,
as it's easy to immediately notice the impact of the optimizer:

```sql
SELECT (("t0"."sepal_length" - 5.843333333333334) * -0.11190585392686306) + 0 + (("t0"."sepal_width" - 3.0573333333333337) * -0.04007948649493375) + (("t0"."petal_length" - 3.7580000000000005) * 0.22864502724212313) + (("t0"."petal_width" - 1.1993333333333336) * 0.6092520541197893) + 1.0000000000000002 AS "variable" FROM "DATA_TABLE" AS "t0"
```

You can notice that `+ 0` operation in the SQL generated by the translator.
Enabling the Optimizer takes care of a few improvements, including getting
rid of that operation which is unecessary:

```sql
SELECT ("t0"."sepal_length" - 5.843333333333334) * -0.11190585392686306 + ("t0"."sepal_width" - 3.0573333333333337) * -0.04007948649493375 + ("t0"."petal_length" - 3.7580000000000005) * 0.22864502724212313 + ("t0"."petal_width" - 1.1993333333333336) * 0.6092520541197893 + 1.0000000000000002 AS "variable" FROM "DATA_TABLE" AS "t0"
```

