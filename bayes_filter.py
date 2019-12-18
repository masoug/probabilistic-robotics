"""Introduction to the Bayes Filter from chapter 2.4 of the Probabilistic Robotics Book"""


class BayesFilter(object):
    """Abstract interface for all recursive bayes filters.
    A recursive bayes filter attempts to estimate the state of a system (robot) based on a sequence of noisy
    measurements and actions. If the distributions of p(measurement | state) and p(state | action, previous state)
    are known, then we can 'reverse' this to estimate p(state | measurement, action) using Bayes Rule.

    Consider a belief distribution p(x3 | z1, z2, z3, u1, u2, u3). Applying bayes rule, we can express it as
    p(z3 | x3, z1, z2, u1, u2, u3) * p(x3 | z1, z2, u1, u2, u3) / p(z3 | z1, z2, u1, u2, u3)

    Starting with p(z3 | z1, z2, u1, u2, u3), we notice that it does not contain any state variables. Hence this value
    stays constant across all state values so we tend to abbreviate this as a normalizer
    n = 1/p(z3 | z1, z2, u1, u2, u3)

    We know p(z3 | x3, z1, z2, u1, u2, u3) because that is the sensor noise model. Now is it the case that the current
    measurement depends on the previous measurements and controls? (i.e. does z3 depend on z1, z2, u1, u2, u3)
    Here, we assume independence between current measurements and past measurements (and controls). Hence, we can
    simplify the sensor distribution to p(z3 | x3).

    p(x3 | z1, z2, u1, u2, u3) corresponds to the prediction, i.e. it estimates the belief distribution before
    incorporating the latest measurement z3. We don't know this distribution directly unless we run a simulation from
    x1 to x3 using z1, z2, u1, u2, u3 every time. But we can expand this expression:
    p(x3 | z1, z2, u1, u2, u3) = integral[ p(x3 | x2, z1, z2, u1, u2, u3) * p(x2 | z1, z2, u1, u2, u3) dx2]
    using the theorem of total probability. We know p(x3 | x2, z1, z2, u1, u2, u3), since that is the state transition
    distribution and, given the independence assumption, the markov chain: p(x3 | x2, u3). We also know
    p(x2 | z1, z2, u1, u2, u3), eliminating u3 since the latest control input does not affect the previous state x2.
    Thus the final expression if p(x3 | z1, z2, u1, u2, u3) = integral[p(x3 | x2, u3) * p(x2 | z1, z2, u1, u2) dx2].
    Notice that p(x2 | z1, z2, u1, u2) is the expression of the previous state belief distribution, which 'closes' the
    loop on the whole recursive part of the bayes filter.

    Now we have a complete formula that maps a previous belief p(x2 | z1, z2, u1, u2) to the next belief
    p(x3 | z1, z2, z3, u1, u2, u3):

    p(x3 | z1, z2, z3, u1, u2, u3) = n * p(z3 | x3) * integral[p(x3 | x2, u3) * p(x2 | z1, z2, u1, u2) dx2]

    These steps are outlined in the predict and measurement update methods below, where subclasses implement their
    own approaches to compute the prediction and updates."""
    def __init__(self, initial_belief):
        # a 'belief' is the set of all possible states and their corresponding probability masses
        self.belief = initial_belief

    def predict(self, control_input):
        raise NotImplementedError('The prediction method is not implemented! The prediction step of a bayes filter'
                                  'applies the control input actions on the previous belief, producing a new belief'
                                  'that reflects what the filter "expects" the corresponding measurement should be.')

    def measurement_update(self, measurement, prediction):
        raise NotImplementedError('The measurement update method is not implemented! The measurement update step of'
                                  'a bayes filter integrates the noisy measurement with the prediction to produce'
                                  'a new belief distribution.')

    def step(self, measurement, control_input):
        """Implements the overall framework for a bayes filter as prescribed in table 2.1 of the probabilistic
        robotics book. """
        prediction = self.predict(control_input)
        new_belief = self.measurement_update(measurement, prediction)
        self.belief = new_belief
        return new_belief


# measurement noise model for the robot's door sensor, as a lookup table RobotDoorSensor[measurement][state]
RobotDoorSensor = {'sense_open': {'is_open': 0.6, 'is_closed': 0.2},
                   'sense_closed': {'is_open': 0.4, 'is_closed': 0.8}}

# manipulator noise model for robot's door pusher, as a lookup table RobotDoorManipulator[state][action][prev_state]
RobotDoorManipulator = {
    'is_open': {
        'push': {'is_open': 1.0, 'is_closed': 0.8},
        'do_nothing': {'is_open': 1.0, 'is_closed': 0.0}
    },
    'is_closed': {
        'push': {'is_open': 0.0, 'is_closed': 0.2},
        'do_nothing': {'is_open': 0.0, 'is_closed': 1.0}
    }}


class RobotDoorFilter(BayesFilter):
    """Example usage of the bayes filter algorithm. Implements example 2.4.2 of the probabilistic robotics book.
    Here, we use the framework of the BayesFilter class to implement a simple door state estimator. This class
    overrides the predict and measurement update steps to explain the theory and intuition of how this filter works."""

    def __init__(self):
        super(RobotDoorFilter, self).__init__({'is_open': 0.5, 'is_closed': 0.5})

    def predict(self, control_input):
        """Implement the prediction step given by sum(x0)[ p(x1 | u, x0) * bel(x0) ]
        Intuitively, we are estimating a new belief (prediction) based on the robot's state transition probabilities
        and the previous belief state probabilities. Notice here that what we're actually doing is applying the theorem
        of total probability, i.e. p(x) = sum(y)[ p(x | y) * p(y)], combining the belief distribution with the
        state transition distribution."""
        prediction = {k: 0.0 for k in self.belief.keys()}

        # for each possible next state...
        for next_state in self.belief.keys():
            # ...sum the probabilities the robot could transition to this next_state...
            for prev_state in self.belief.keys():
                # ...given by the product of the probability the robot is in some previous state (belief)
                # and the probability that the robot will transition to the next_state
                prediction[next_state] += RobotDoorManipulator[next_state][control_input][prev_state] \
                                          * self.belief[prev_state]

        # you're then left with a new belief that reflects the application of the control input on the previous
        # belief distribution
        return prediction

    def measurement_update(self, measurement, prediction):
        """Implement the measurement update step given by bel(x1) = n * p(z | x1) * prediction(x1)
        Intuitively, this phase 'corrects' the prediction based on a noisy measurement of the state."""
        new_belief = {}

        # compute the new probabilities for each next_state, by multiplying the sensor measurement probability with the
        # predicted belief distribution...
        for next_state in self.belief.keys():
            new_belief[next_state] = RobotDoorSensor[measurement][next_state] * prediction[next_state]

        # ...then normalize the probabilities. This keeps the belief distribution as a probability density function
        # since the products may sum greater than 1.
        normalization = sum(new_belief.values())
        for state in self.belief.keys():
            new_belief[state] /= normalization

        return new_belief


def main():
    """The Bayes Filter is the fundamental algorithm behind estimating a belief distribution from control and
    measurement data. It operates in two phases, a prediction step and then a measurement update step. Used
    recursively, the bayes filter can iteratively update the belief distribution for each new measurement and control
    action.

    Here we simulate a robot trying to estimate the state of a door its perceiving and actuating. We've defined the
    noise models for both the robot's door sensor and its door actuator. Using these distributions, we can analyze the
    stream of information from the robot's sensors and actuators to generate a belief distribution for the current
    state of the door. The belief distribution in this case has two states, is_open and is_closed, and each state
    is assigned a probability value indicating the likelihood the door is in that state. This belief is updated over
    time as the robot cycles between sensing and applying an action, with the hope that the belief distribution will
    converge to one state of the door."""

    door_filter = RobotDoorFilter()
    print(f'Door filter initial belief: {door_filter.belief}')

    print(f"  Sense open and did nothing: {door_filter.step('sense_open', 'do_nothing')}")
    print(f"  Sense open and pushed: {door_filter.step('sense_open', 'push')}")


if __name__ == '__main__':
    main()
