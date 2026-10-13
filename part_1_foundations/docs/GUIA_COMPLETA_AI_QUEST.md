# Guia completa de AI Quest - Parte 1

Esta guia explica la primera parte del proyecto, ubicada en `part_1_foundations/`: que archivos hay, como se conectan, que significa cada capa, que hace cada neurona, cual es la matematica detras, y como usar `neural_network.py`, `show_digit.py` y `compare_versions.py` para explorar diferentes versiones del detector de digitos.

La idea central del proyecto es construir una red neuronal pequena desde cero, sin usar TensorFlow ni PyTorch. Usamos `scikit-learn` solo para cargar una base de datos real de digitos escritos a mano. La red y sus neuronas son nuestras.

## Mapa general

El proyecto tiene estos archivos importantes:

```text
understand-ai/            # repository root
  README.md
  part_1_foundations/
    neural_network.py
    single_neuron_digit_detector.py
    compare_versions.py
    show_digit.py
    show.py
    docs/
      GUIA_COMPLETA_AI_QUEST.md
      part_i_ground_notes.md
      ai_quest_en.md
    assets/
      digits/
        digit_0_examples.png
        digit_1_examples.png
        ...
      plots/
        and_neuron_boundary.png
        xor_trained_network.png
        ...
  part_3_visualizer/
    ...
```

Los tres archivos principales son:

```text
neural_network.py
```

Es el archivo principal para entender la red. Tiene:

- La clase `Neuron`.
- La clase `Layer`.
- La clase `NeuralNetwork`.
- La funcion `sigmoid`.
- Funciones de entrenamiento basicas.
- Un detector de digitos jerarquico hecho a mano.
- Un menu para analizar ejemplos.

```text
show_digit.py
```

Sirve para ver visualmente muchas imagenes del dataset. Por ejemplo, todos los digitos etiquetados como `5`.

```text
compare_versions.py
```

Sirve para comparar varias versiones del detector:

- Version baseline.
- Version con strokes.
- Version con patches.
- Versiones con imagenes ampliadas.
- Versiones con mas neuronas en la primera capa.

Tambien mide:

- Cuantas predicciones fueron correctas.
- Accuracy total.
- Tiempo de ejecucion.
- Mejor digito.
- Peor digito.
- Accuracy por digito.

## La base de datos de digitos

Usamos:

```python
from sklearn.datasets import load_digits
digits = load_digits()
```

Este dataset trae imagenes pequenas de digitos escritos a mano.

Cada imagen es de:

```text
8 x 8 pixeles
```

Eso significa que una imagen tiene:

```text
64 pixeles
```

Cada pixel es un numero de `0` a `16`:

```text
0  = blanco / apagado
16 = negro / muy activo
```

Por ejemplo, un digito no se guarda como una imagen PNG directamente. Se guarda como una matriz de numeros:

```text
[
  [0, 0, 5, 13, 9, 1, 0, 0],
  [0, 0, 13, 15, 10, 15, 5, 0],
  ...
]
```

La etiqueta real del digito se guarda aparte:

```python
digits.target[index]
```

Si `digits.target[7] == 7`, significa que la imagen en `digits.images[7]` representa un `7`.

## Como mostramos los digitos en consola

En `neural_network.py`, la funcion:

```python
def pixel_to_shade(pixel):
    shades = " .:-=+*#%@"
    index = round((pixel / 16) * (len(shades) - 1))
    return shades[index]
```

convierte un pixel numerico en un caracter visual.

La escala es:

```text
espacio = pixel muy claro
.       = un poco mas oscuro
:
-
=
+
*
#
%
@       = pixel muy oscuro
```

Entonces esto:

```text
|    --##++..    |
|    ##%%**%%--  |
|  ::%%..  **==  |
```

es una forma de ver en terminal una imagen de `8 x 8`.

Importante: la red no recibe esos caracteres. Los caracteres son solo para que nosotros podamos ver el digito. La red recibe numeros.

## La neurona

La clase principal es:

```python
class Neuron:
```

Una neurona tiene:

```python
self.weights
self.bias
self.last_inputs
self.last_output
```

Los `weights` son los pesos.

El `bias` es un ajuste extra, como un umbral.

`last_inputs` y `last_output` guardan lo ultimo que paso por la neurona. Eso sirve para entrenamiento con `backward`, aunque nuestro detector actual de digitos todavia no aprende.

## Que hace forward

El metodo mas importante es:

```python
def forward(self, inputs):
    z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
    self.last_inputs = inputs
    self.last_output = sigmoid(z)
    return self.last_output
```

Matematicamente:

```text
z = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn + b
```

Luego:

```text
output = sigmoid(z)
```

Donde:

```text
x = input
w = weight
b = bias
z = suma ponderada
```

La neurona primero mezcla los inputs con pesos. Luego pasa el resultado por `sigmoid`.

## La funcion sigmoid

En el codigo:

```python
def sigmoid(z):
    return 1 / (1 + math.exp(-z))
```

Matematicamente:

```text
sigmoid(z) = 1 / (1 + e^(-z))
```

La sigmoid convierte cualquier numero en un valor entre `0` y `1`.

Ejemplos:

```text
z muy negativo -> output cerca de 0
z = 0          -> output = 0.5
z muy positivo -> output cerca de 1
```

En nuestro proyecto, podemos pensar:

```text
output cerca de 0 = neurona apagada
output cerca de 1 = neurona prendida
```

Pero ojo: "prendida" no es una categoria absoluta. Nosotros usamos un umbral, por ejemplo `0.5`, para decir:

```text
activation >= 0.5 -> ON
activation < 0.5  -> off
```

## Capas

Una capa es un grupo de neuronas.

En `neural_network.py`:

```python
class Layer:
    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]
```

Si una capa tiene 8 neuronas, recibe los mismos inputs y produce 8 outputs:

```text
inputs -> [neurona 1, neurona 2, ..., neurona 8] -> 8 activaciones
```

## Red neuronal general

Tambien existe:

```python
class NeuralNetwork:
```

Esta clase permite crear redes como:

```python
NeuralNetwork([2, 4, 1])
```

Eso significa:

```text
2 inputs -> 4 neuronas ocultas -> 1 output
```

Su `forward` hace:

```python
activations = inputs
for layer in self.layers:
    activations = layer.forward(activations)
return activations
```

O sea:

```text
inputs
-> capa 1
-> capa 2
-> capa 3
-> output final
```

Esta parte es una red neuronal general. La dejamos porque sirve para aprender el concepto completo. Pero el detector de digitos actual usa una estructura hecha a mano, mas didactica.

## Entrenamiento

El archivo tambien tiene:

```python
train_single_neuron(...)
NeuralNetwork.backward(...)
NeuralNetwork.train(...)
```

Estas funciones son para entrenamiento real con ajuste de pesos.

Pero la parte actual de digitos:

```text
NO entrena todavia
```

Los pesos estan puestos a mano o salen de una referencia simple.

Eso es intencional: queriamos primero entender como fluye la informacion.

## La idea estilo 3Blue1Brown

La intuicion que buscamos fue:

```text
pixeles
-> trazos pequenos
-> partes mas grandes
-> digito final
```

Algo como:

```text
pixeles oscuros arriba -> neurona top se activa
pixeles oscuros a la derecha -> neurona upper_right se activa
top + upper_right + center_vertical -> parece un 7
```

En redes reales, esas capas aprenden patrones automaticamente. En nuestro proyecto, los patrones se definen a mano para que se puedan ver y explicar.

## Baseline

`baseline` significa la version base.

En nuestro caso, el baseline es:

```text
el detector jerarquico original de neural_network.py
```

Este baseline usa:

```text
8 neuronas en la primera capa
10 neuronas en la salida
```

Primera capa:

```text
top
middle
bottom
upper_left
lower_left
upper_right
lower_right
center_vertical
```

Salida:

```text
neurona para 0
neurona para 1
neurona para 2
...
neurona para 9
```

El flujo es:

```text
64 pixeles
-> 8 detectores de trazos
-> 10 neuronas de salida
-> prediccion
```

## Strokes

`strokes` significa trazos.

Un stroke es una region con significado humano:

```text
top          = parte de arriba
middle       = parte central
bottom       = parte de abajo
upper_left   = lado superior izquierdo
lower_left   = lado inferior izquierdo
upper_right  = lado superior derecho
lower_right  = lado inferior derecho
center_vertical = columna central
```

Por ejemplo, para un `0`, esperamos:

```text
top
bottom
upper_left
lower_left
upper_right
lower_right
```

Pero no esperamos:

```text
middle
center_vertical
```

Para un `8`, esperamos casi todo:

```text
top
middle
bottom
upper_left
lower_left
upper_right
lower_right
```

## Como se crea una neurona stroke

En `neural_network.py`:

```python
def make_stroke_detector(region, threshold=0.32, strength=10):
```

Esta funcion crea una neurona que mira solo una region de la imagen.

Ejemplo: si la region es `top`, la neurona solo mira pixeles de arriba.

El codigo pone:

```python
detector.weights = [0] * 64
```

Eso significa que al principio ningun pixel importa.

Luego:

```python
for row, col in region:
    detector.weights[pixel_index(row, col)] = strength / len(region)
```

Solo los pixeles dentro de esa region reciben peso positivo.

Entonces la neurona calcula algo equivalente a:

```text
z = strength * promedio_de_pixeles_en_la_region - strength * threshold
```

Porque el bias es:

```python
detector.bias = -strength * threshold
```

Matematicamente:

```text
activation = sigmoid(strength * (average_region_intensity - threshold))
```

Eso es super importante.

Significa:

```text
si el promedio de la region es mayor que threshold -> la neurona se activa
si el promedio de la region es menor que threshold -> la neurona se apaga
```

## Threshold

`threshold` es el umbral.

Ejemplo:

```python
threshold = 0.32
```

Como los pixeles se normalizan de `0..16` a `0..1`, un threshold de `0.32` significa:

```text
la region necesita tener suficiente tinta para prenderse
```

Si bajas threshold:

```python
threshold = 0.18
```

la neurona se prende mas facil.

Efecto:

```text
mas detecciones
mas ruido
mas falsos positivos
```

Si subes threshold:

```python
threshold = 0.48
```

la neurona exige mas oscuridad.

Efecto:

```text
menos detecciones
puede ignorar trazos suaves
puede reducir ruido
```

## Strength

`strength` controla que tan fuerte responde la neurona.

Ejemplo:

```python
strength = 10
```

La formula aproximada es:

```text
activation = sigmoid(strength * (average - threshold))
```

Si `strength` es alto:

```text
la sigmoid se vuelve mas agresiva
las activaciones se acercan mas rapido a 0 o 1
```

Si `strength` es bajo:

```text
las activaciones son mas suaves
hay mas valores intermedios como 0.35, 0.48, 0.62
```

## Segunda capa en strokes

Despues de detectar strokes, tenemos neuronas de salida.

Cada neurona de salida representa un digito:

```text
output neuron 0
output neuron 1
...
output neuron 9
```

En `neural_network.py`, `DIGIT_STROKES` dice que trazos espera cada digito.

Ejemplo:

```python
3: ["top", "middle", "bottom", "upper_right", "lower_right"]
```

Eso significa:

```text
para que parezca un 3, queremos:
arriba
medio
abajo
derecha arriba
derecha abajo
```

La neurona final del `3` recibe las activaciones de la primera capa.

Si un stroke esperado esta activo, suma puntos.

Si un stroke no esperado esta activo, resta puntos.

## Patches

`patches` significa pedacitos o cuadritos.

En vez de decir manualmente:

```text
top
middle
bottom
```

dividimos la imagen en una cuadricula.

Por ejemplo:

```python
grid = 4
```

divide la imagen en:

```text
4 x 4 = 16 patches
```

Cada patch es una neurona de la primera capa.

Si tenemos una imagen `8 x 8` y `grid = 4`, cada patch cubre aproximadamente:

```text
2 x 2 pixeles
```

Si tenemos una imagen `16 x 16` y `grid = 8`, cada patch tambien cubre:

```text
2 x 2 pixeles
```

Pero ahora hay mas patches:

```text
8 x 8 = 64 neuronas
```

## Numero de neuronas

En versiones `patches`, el numero de neuronas de la primera capa depende de:

```python
grid
```

Formula:

```text
neuronas primera capa = grid * grid
```

Ejemplos:

```text
grid = 4  -> 16 neuronas
grid = 6  -> 36 neuronas
grid = 8  -> 64 neuronas
grid = 10 -> 100 neuronas
grid = 16 -> 256 neuronas
```

La capa de salida sigue teniendo:

```text
10 neuronas
```

una para cada digito.

## Upscale

`upscale` agranda la imagen.

El dataset original es:

```text
8 x 8
```

Si:

```python
upscale = 2
```

la imagen se vuelve:

```text
16 x 16
```

El metodo que usamos es nearest-neighbor simple:

```text
cada pixel se copia en un bloque mas grande
```

Ejemplo conceptual:

```text
5
```

con upscale 2 se vuelve:

```text
5 5
5 5
```

Eso no inventa informacion nueva. Solo agranda la imagen.

Por eso:

```text
upscale solo no necesariamente mejora accuracy
```

Pero puede ayudar si tambien aumentamos `grid`, porque permite crear mas regiones/pedazos.

## Como funcionan patches matematicamente

Cada patch mira una region.

Igual que strokes:

```text
activation_patch = sigmoid(strength * (average_patch - threshold))
```

La diferencia es que el patch no tiene nombre semantico como `top`.

Tiene nombre geometrico:

```text
patch_0_0
patch_0_1
patch_0_2
...
patch_7_7
```

`patch_0_0` significa:

```text
fila de patches 0, columna de patches 0
```

Es la esquina superior izquierda.

`patch_7_7` en un grid 8 es:

```text
esquina inferior derecha
```

## Baseline vs strokes vs patches

Resumen:

```text
baseline
  usa lo mismo que neural_network.py
  8 strokes hechos a mano
  muy explicable
  no necesariamente preciso

strokes
  usa regiones tipo segmentos
  tambien es explicable
  puedes cambiar threshold, strength, pesos y upscale

patches
  usa muchos cuadritos
  tiene mas neuronas
  menos humano, mas geometrico
  puede capturar mas detalle
  puede ser mas lento
```

## Como se decide la prediccion

Despues de la primera capa:

```text
pixeles -> activaciones de strokes/patches
```

La salida calcula un score para cada digito:

```text
output_0
output_1
output_2
...
output_9
```

Luego se elige:

```text
prediccion = digito cuya neurona de salida tuvo mayor activacion
```

Importante:

```text
la neurona ganadora no siempre esta muy prendida
```

Por ejemplo:

```text
digit 5: 0.121 <- strongest output / prediction
```

Eso significa:

```text
la red eligio 5 porque fue el mayor valor
pero 0.121 es una activacion baja
entonces la red no esta segura
```

En cambio:

```text
digit 7: 0.992 ON
```

significa que la neurona del `7` esta muy activada.

## Output ON vs prediccion

El codigo usa dos ideas:

```text
prediction = neurona con score mas alto
ON = neurona con score >= 0.5
```

Puede pasar:

```text
prediction: output neuron 5
pero ninguna neurona paso 0.5
```

Eso significa:

```text
5 fue la opcion menos mala
```

No significa que la red este segura.

## Prototype patches

En `compare_versions.py`, las versiones `patches` usan una idea llamada prototipo.

No entrenan con gradient descent.

Hacen esto:

1. Buscan el primer ejemplo de cada digito.
2. Pasan ese ejemplo por la primera capa de patches.
3. Guardan el patron de activaciones.
4. Usan ese patron como referencia del digito.

Por ejemplo:

```text
primer 7 del dataset -> activaciones de patches -> prototipo del 7
```

Luego, para una imagen nueva:

```text
imagen nueva -> activaciones de patches -> comparar con prototipo de cada digito
```

La neurona de salida del `7` tiene pesos basados en el prototipo del `7`.

En codigo:

```python
neuron.weights = [(activation * 2) - 1 for activation in prototype]
```

Si en el prototipo un patch estaba muy activo:

```text
activation cerca de 1 -> weight cerca de +1
```

Si en el prototipo un patch estaba apagado:

```text
activation cerca de 0 -> weight cerca de -1
```

Matematicamente:

```text
weight_i = 2 * prototype_i - 1
```

Entonces:

```text
prototype_i = 1.0 -> weight_i = 1.0
prototype_i = 0.5 -> weight_i = 0.0
prototype_i = 0.0 -> weight_i = -1.0
```

Es una forma rudimentaria de decir:

```text
si este patch era importante en el ejemplo original, premialo
si no era importante, castigalo
```

## Prototype bias

En patches:

```python
prototype_bias
```

controla que tan exigente es la neurona de salida.

El bias se calcula asi:

```python
neuron.bias = -prototype_bias * sum(max(weight, 0) for weight in neuron.weights)
```

Eso significa:

```text
a mayor prototype_bias, mas dificil que la neurona se active
```

Si bajas:

```python
prototype_bias = 0.35
```

la salida se prende mas facil.

Si subes:

```python
prototype_bias = 0.75
```

la salida exige mas coincidencias.

## VERSION_CONFIGS

En `compare_versions.py`, arriba esta:

```python
VERSION_CONFIGS = [
    ...
]
```

Cada bloque es una version que se va a comparar.

Ejemplo:

```python
{
    "name": "patches 8x8 on 16x16",
    "kind": "patches",
    "grid": 8,
    "upscale": 2,
    "threshold": 0.48,
    "strength": 8,
    "prototype_bias": 0.75,
}
```

Esto significa:

```text
nombre: patches 8x8 on 16x16
tipo: patches
grid: 8 x 8 = 64 neuronas en primera capa
upscale: imagen original 8x8 se vuelve 16x16
threshold: patches exigen intensidad 0.48
strength: respuesta de la sigmoid
prototype_bias: exigencia de la salida
```

## Como cambiar el numero de neuronas

En una version `patches`, cambia:

```python
"grid": 8
```

Formula:

```text
neuronas = grid * grid
```

Si quieres probar 100 neuronas:

```python
"grid": 10
```

Si quieres probar 256 neuronas:

```python
"grid": 16
```

Ojo:

```text
mas neuronas = mas lento
mas neuronas = mas detalle
mas neuronas no siempre = mejor accuracy
```

## Como agregar una version nueva

Copia un bloque en `VERSION_CONFIGS`.

Ejemplo:

```python
{
    "name": "patches 6x6 on 16x16",
    "kind": "patches",
    "grid": 6,
    "upscale": 2,
    "threshold": 0.25,
    "strength": 8,
    "prototype_bias": 0.55,
}
```

Esto crea:

```text
36 neuronas de patches
imagen 16x16
threshold 0.25
```

Luego corre:

```bash
.venv\Scripts\python.exe part_1_foundations/compare_versions.py --no-menu
```

## Como editar parametros desde el menu

Corre:

```bash
.venv\Scripts\python.exe part_1_foundations/compare_versions.py --menu
```

Luego escoge:

```text
3. Edit version parameters and compare again
```

El menu te muestra los parametros actuales.

Puedes cambiar:

```text
grid
upscale
threshold
strength
prototype_bias
positive_weight
negative_weight
bias_scale
```

Despues reconstruye las versiones y vuelve a medir los `1797` samples.

## Como leer la tabla de comparacion

La tabla se ve asi:

```text
id | version               | correct  | accuracy | time    | best digit | worst digit
---+-----------------------+----------+----------+---------+------------+------------
0  | baseline from neural_network.py | 758/1797 | 42.18%   | 0.1756s | 6 (90.1%)  | 4 (0.6%)
```

`id`:

```text
numero de version
```

Lo usas en el menu para elegir modelo.

`correct`:

```text
predicciones correctas / total de imagenes
```

`accuracy`:

```text
correct / total
```

`time`:

```text
cuanto tardo en evaluar todos los samples
```

`best digit`:

```text
digito que predijo mejor
```

`worst digit`:

```text
digito que predijo peor
```

## Accuracy por digito

Si corres:

```bash
.venv\Scripts\python.exe part_1_foundations/compare_versions.py --details --no-menu
```

ves una tabla por digito.

Ejemplo:

```text
digit | correct | accuracy
------+---------+---------
0     | 174/178 | 97.75%
1     | 73/182  | 40.11%
...
```

Esto sirve para entender:

```text
esta version reconoce bien el 0
pero falla mucho con el 9
```

## show_digit.py

Este archivo no explica capas. Solo genera una imagen con todos los ejemplos de un digito.

Ejemplo:

```bash
.venv\Scripts\python.exe part_1_foundations/show_digit.py 5
```

Genera:

```text
part_1_foundations/assets/digits/digit_5_examples.png
```

Sirve para ver variacion real en el dataset:

```text
no todos los 5 se ven igual
algunos parecen 3
algunos parecen 6
algunos estan inclinados
```

Eso explica por que nuestro detector manual falla.

## show.py

`show.py` es una version vieja/simple:

```python
plt.matshow(digits.images[73], cmap="gray")
plt.show()
```

Solo muestra una imagen fija.

Actualmente `show_digit.py` es mas util.

## Por que no tenemos accuracy alta

Porque no estamos entrenando la red.

Estamos haciendo heuristicas:

```text
strokes hechos a mano
patches comparados contra un solo ejemplo prototipo
```

Una red real aprenderia pesos usando muchos ejemplos.

Nosotros estamos diciendo:

```text
si tiene tinta arriba, probablemente top
si tiene tinta abajo, probablemente bottom
si parece el primer ejemplo de 7, probablemente 7
```

Eso es didactico, pero limitado.

## Que seria aprender de verdad

Aprender de verdad seria:

1. Inicializar pesos.
2. Pasar una imagen con `forward`.
3. Comparar output con target real.
4. Calcular error.
5. Usar `backward` para ajustar pesos.
6. Repetir miles de veces.

El objetivo seria que la red encuentre sola:

```text
que pixeles importan
que patches importan
que combinaciones predicen cada digito
```

En ese caso, ya no escribirias `DIGIT_STROKES` a mano.

La red aprenderia pesos automaticamente.

## Relacion con 3Blue1Brown

En el video, la idea conceptual es:

```text
primera capa: detecta trazos/patrones simples
segunda capa: combina trazos en formas
tercera/salida: combina formas en digitos
```

Nuestro proyecto imita esa idea, pero de forma manual:

```text
pixels -> strokes/patches -> output digit neurons
```

En una red entrenada, los strokes/patches no los nombramos nosotros. La red descubre patrones mediante entrenamiento.

## Comandos principales

Demo interactivo de `neural_network.py`:

```bash
.venv\Scripts\python.exe part_1_foundations/neural_network.py
```

Analizar un indice exacto:

```bash
.venv\Scripts\python.exe part_1_foundations/neural_network.py --index 42
```

Analizar ejemplos de un digito:

```bash
.venv\Scripts\python.exe part_1_foundations/neural_network.py --digit 7 --limit 3
```

Medir accuracy rudimentaria:

```bash
.venv\Scripts\python.exe part_1_foundations/neural_network.py --accuracy 100
```

Comparar versiones:

```bash
.venv\Scripts\python.exe part_1_foundations/compare_versions.py --no-menu
```

Comparar con detalle por digito:

```bash
.venv\Scripts\python.exe part_1_foundations/compare_versions.py --details --no-menu
```

Comparar y abrir menu:

```bash
.venv\Scripts\python.exe part_1_foundations/compare_versions.py --menu
```

Generar imagen con todos los ejemplos de un digito:

```bash
.venv\Scripts\python.exe part_1_foundations/show_digit.py 5
```

## Como estudiar el proyecto

Orden recomendado:

1. Lee `Neuron.forward`.
2. Entiende `sigmoid`.
3. Corre `neural_network.py --index 0`.
4. Mira que strokes se activan.
5. Mira la output layer.
6. Corre `neural_network.py --digit 7 --limit 3`.
7. Observa como varian imagenes del mismo digito.
8. Corre `compare_versions.py --no-menu`.
9. Cambia `grid`.
10. Cambia `threshold`.
11. Cambia `prototype_bias`.
12. Compara resultados.

## Preguntas clave para experimentar

Cuando cambies parametros, preguntate:

```text
Subio accuracy?
Bajo tiempo?
Mejoro algun digito?
Empeoro otro?
La red esta mas segura o solo mas agresiva?
Mas neuronas realmente ayudaron?
El modelo se volvio mas lento?
```

## Resumen mental final

El proyecto hace esto:

```text
1. Carga imagenes reales de digitos.
2. Convierte pixeles de 0..16 a 0..1.
3. Primera capa detecta strokes o patches.
4. Segunda/salida combina activaciones.
5. La neurona de salida mas activa es la prediccion.
6. compare_versions.py prueba varias configuraciones.
7. Las tablas muestran cual version fue mejor, peor y mas rapida.
```

La frase clave:

```text
No estamos entrenando todavia; estamos construyendo detectores manuales para entender como fluye la informacion en una red neuronal.
```
