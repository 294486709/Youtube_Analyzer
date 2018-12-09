import sys, random, math, pygame, os
from pygame.locals import *


def dashboard(conclution):
    background_image = './bg.png'
    pointer_image = './pointer.png'

    pygame.init()

    SCREEN_SIZE = (1444, 708)

    screen = pygame.display.set_mode(SCREEN_SIZE, 0, 32)

    pygame.display.set_caption('Political leaning dashboard')

    background = pygame.image.load(background_image).convert_alpha()
    pointer = pygame.image.load(pointer_image).convert_alpha()

    width, height = background.get_size()
    width_2, height_2 = pointer.get_size()

    pointer = pygame.transform.smoothscale(pointer, (width_2 * 2, height_2 * 2))

    font = pygame.font.Font(None, 18)

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            sys.exit()

        screen.blit(background, (0, 0))
        if conclution == 'neutral':
            screen.blit(pointer, ((width-width_2*2)/2 , height - 616))
        elif conclution == 'left':
            pointer_rotate = pygame.transform.rotate(pointer, 60)
            screen.blit(pointer_rotate, ((width - width_2 * 2) / 2 - 480, height - 366 + 7))
        elif conclution == 'right':
            pointer_rotate_2 = pygame.transform.rotate(pointer, -60)
            screen.blit(pointer_rotate_2, ((width - width_2 * 2) / 2 - 4, height - 360))

        pygame.display.update()

def main():
    a = input('input:\n')
    dashboard(a)


if __name__ == '__main__':
    main()
