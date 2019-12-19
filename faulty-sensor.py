"""Applying bayes filter to the faulty sensor problem in chapter 2 of Probabilistic Robotics Book"""
from bayes_filter import HistogramFilter


"""
 Conditional Probability Table (in frequencies):
               Faulty    Working
        Near:      3        97    100
        Far:       0       200    200
                   3       297    300
"""

if __name__ == '__main__':
    # p(measurement | faulty) = sensor[measurement][faulty]
    sensor = {
        'sense_near': {'is_faulty': 1.0, 'is_working': 97./297.},
        'sense_far':  {'is_faulty': 0.0, 'is_working': 200./297.}
    }

    # action[state][action][prev_state], static transition table, we do nothing
    action = {
        'is_faulty': {
            'do_nothing': {'is_faulty': 1.0, 'is_working': 0.0}
        },
        'is_working': {
            'do_nothing': {'is_faulty': 0.0, 'is_working': 1.0}
        }
    }

    # we seed the initial belief using the marginal probabilities of the sensor state
    initial_belief = {'is_faulty': 0.01, 'is_working': 0.99}
    print('Assuming no action is performed and the sensor always reports "near"')
    print(f'Using initial belief {initial_belief}')
    sensor_filter = HistogramFilter(sensor, action, initial_belief)
    for t in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        print(f'At time t={t}, the belief yields {sensor_filter.step("sense_near", "do_nothing")}')
