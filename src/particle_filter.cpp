/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	std::default_random_engine generator;
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	num_particles = 1000;

	for (int i = 0; i < num_particles; i++){
		Particle p;
		p.x = dist_x(generator);
		p.y = dist_y(generator);
		p.theta = dist_theta(generator);
        p.weight = 1;
        weights.push_back(1);
		particles.push_back(p);
	}

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine generator;
	for (int i = 0; i < num_particles; i++){
        double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		std::normal_distribution<double> dist_x(x, std_pos[0]);
		std::normal_distribution<double> dist_y(y, std_pos[1]);
		std::normal_distribution<double> dist_theta(theta, std_pos[2]);
		double noisy_x = dist_x(generator);
		double noisy_y = dist_y(generator);
		double noisy_theta = dist_theta(generator);
		particles[i].x = noisy_x +
				((velocity/yaw_rate)*(std::sin(noisy_theta + yaw_rate*delta_t) - std::sin(noisy_theta)));
		particles[i].y = noisy_y +
						 ((velocity/yaw_rate)*(std::cos(noisy_theta) - std::cos(noisy_theta + yaw_rate*delta_t)));
		particles[i].theta = noisy_theta + yaw_rate * delta_t;
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int obs_i = 0; obs_i < observations.size(); obs_i++){
		LandmarkObs current_obs = observations[obs_i];
		double closest_dist = std::numeric_limits<double>::max();
		for (int pred_i = 0; pred_i < predicted.size(); pred_i++){
			LandmarkObs current_pred = predicted[pred_i];
			double d = dist(current_obs.x, current_obs.y, current_pred.x, current_pred.y);
			if (d < closest_dist){
				closest_dist = d;
				current_obs.id = current_pred.id;
			}
		}
		observations[obs_i] = current_obs;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account 
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	double n_cost = 1.0 / (2*M_PI*std_landmark[0]*std_landmark[1]);
	double n_denom_1 = 2*pow(std_landmark[0], 2);
	double n_denom_2 = 2*pow(std_landmark[1], 2);
	for (int i = 0; i < num_particles; i++){
		std::vector<LandmarkObs> predicted;
		double particles_probability = 1;
		Particle p = particles[i];
		// select landmarks in sensor range for this particle. And keep in map coordinates;
		for (int landmark = 0; landmark < map_landmarks.landmark_list.size(); landmark++){
			Map::single_landmark_s l = map_landmarks.landmark_list[landmark];
            LandmarkObs l_predicted;
			double d = dist(p.x, p.y, l.x_f, l.y_f);
			if (d <= sensor_range){
				l_predicted.x = l.x_f;
				l_predicted.y = l.y_f;
				l_predicted.id = l.id_i;
				predicted.push_back(l_predicted);
			}
		}
		// Convert observations to map coordinates relative to this particle.
		std::vector<LandmarkObs> obs_converted;
		for (int obs = 0; obs < observations.size(); obs++){
			LandmarkObs o;
			o.x = o.x * std::cos(p.theta) + o.y * std::sin(p.theta) + p.x;
			o.y = o.x * std::sin(p.theta) + o.y * std::cos(p.theta) + p.y;
			obs_converted.push_back(o);
		}
		// associate landmarks to observations
		dataAssociation(predicted, obs_converted);
		// calc normal prob for all observations
		for (int c = 0; c < obs_converted.size(); c++){
			LandmarkObs current_observation = obs_converted[c];
			Map::single_landmark_s closest_landmark = map_landmarks.landmark_list[current_observation.id];
            double n_exp = -1 * ((pow(current_observation.x-closest_landmark.x_f,2)/n_denom_1) +
					(pow(current_observation.y-closest_landmark.y_f,2)/n_denom_2));
			double probability = n_cost * exp(n_exp);
			particles_probability = particles_probability * probability;
		}
		// update weights
		weights[i] = particles_probability;
	}
    // normalize weights
	double sum_weights = 0;
	for (int i; i < num_particles; i++){
		sum_weights = sum_weights + weights[i];
	}
	for (int i; i < num_particles; i++){
		double norm_weight = weights[i] / sum_weights;
		particles[i].weight = norm_weight;
		weights[i] = norm_weight;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::vector<Particle> new_particles;
	std::random_device rd;  //Will be used to obtain a seed for the random number engine
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0, 1);
	double beta = 0;
	int index = int(dis(gen) * num_particles);
	double mw = 0;
	for (int i = 0; i < num_particles; i++){
		if (weights[i] > mw){
			mw = weights[i];
		}
	}
	for (int i = 0; i < num_particles; i++){
		beta += dis(gen) * 2.0 * mw;
		while (beta > weights[index]){
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		new_particles.push_back(particles[index]);
	}
	particles = new_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
