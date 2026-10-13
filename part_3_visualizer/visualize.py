"""Interactive, numerical view of the actual Part I network learning XOR."""
import argparse
import random
from pathlib import Path

from part_1_foundations.neural_network import NeuralNetwork

DATA = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]


def run(snapshot=None):
    import pygame
    pygame.init()
    surface = pygame.display.set_mode((1050, 700))
    pygame.display.set_caption('Part III | Part I XOR network')
    font = pygame.font.Font(None, 23)
    random.seed(7)
    network = NeuralNetwork([2, 2, 1])
    clock = pygame.time.Clock()
    selected, epochs, frame = 0, 0, 0
    training, running = True, True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    training = not training
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    selected = event.key - pygame.K_1
        if training:
            for _ in range(50):
                for inputs, target in DATA:
                    network.forward(inputs)
                    network.backward(target, 0.5)
                epochs += 1
        inputs, target = DATA[selected]
        outputs = network.forward(inputs)
        values = [inputs] + [[n.last_output for n in layer.neurons] for layer in network.layers]
        positions = [[(150 + layer * 350, 230 + node * 200) for node in range(len(row))]
                     for layer, row in enumerate(values)]
        surface.fill((16, 22, 34))

        def label(text, x, y, color=(225, 235, 242)):
            surface.blit(font.render(text, True, color), (x, y))

        label('PART III / Watch the Part I network learn XOR', 40, 30)
        label('1–4: choose input | SPACE: pause training | ESC: exit', 40, 65)
        label(f'Epoch {epochs} | input {inputs} | target {target[0]} | output {outputs[0]:.4f}', 40, 110)
        for layer_index, layer in enumerate(network.layers):
            for j, neuron in enumerate(layer.neurons):
                end = positions[layer_index + 1][j]
                for i, weight in enumerate(neuron.weights):
                    start = positions[layer_index][i]
                    color = (80, 185, 240) if weight >= 0 else (249, 126, 107)
                    pygame.draw.line(surface, color, start, end, min(7, 1 + int(abs(weight))))
                    phase = (frame * 0.02) % 1
                    point = tuple(int(a + (b - a) * phase) for a, b in zip(start, end))
                    pygame.draw.circle(surface, (250, 240, 170), point, 4)
                    label(f'w={weight:+.2f}', int(start[0] * .75 + end[0] * .25),
                          int(start[1] * .75 + end[1] * .25) - 20)
                label(f'b={neuron.bias:+.2f}', end[0] - 35, end[1] + 44)
        for row, row_positions in zip(values, positions):
            for activation, point in zip(row, row_positions):
                pygame.draw.circle(surface, (35, 80 + int(activation * 150), 150), point, 34)
                label(f'{activation:.3f}', point[0] - 23, point[1] - 8)
        label('Blue: positive weight | Coral: negative weight | Node label: activation', 40, 585)
        label('a = sigmoid(sum(w * input) + bias). Backpropagation updates these same weights.', 40, 625)
        pygame.display.flip()
        if snapshot:
            Path(snapshot).parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(surface, snapshot)
            running = False
        frame += 1
        clock.tick(30)
    pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', help='Save one frame and exit; SDL dummy driver supports headless QA')
    run(parser.parse_args().snapshot)
