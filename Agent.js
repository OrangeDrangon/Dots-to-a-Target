class Agent {
    constructor(x, y, weights) {
        this.x = x;
        this.y = y;
        this.velocity = [0, 0];
        this.dead = false;
        this.brain = tf.sequential();
        this.brain.add(tf.layers.dense({ units: 8, inputShape: [4], activation: 'sigmoid' }));
        this.brain.add(tf.layers.dense({ units: 4, activation: 'sigmoid' }));
        this.brain.compile({
            optimizer: tf.train.sgd(.5),
            loss: tf.losses.meanSquaredError
        });
        if (weights) {
            this.brain.setWeights(weights);
            for (let i = 0; i < weights.length; i++) {
                weights[i].dispose();
            }
        }
        var d = new Date();
        this.deathTime = d.getTime() + 30 * 1000;
    }
    move() {
        tf.tidy(() => {
            const inputs = tf.tensor2d([[this.x, this.y, this.velocity[0], this.velocity[1]]])
            const outputs = this.brain.predict(inputs);
            this.velocity = outputs.dataSync();
            if (this.velocity[2] <= .5) {
                this.velocity[0] *= -1;
            }
            if (this.velocity[3] <= .5) {
                this.velocity[1] *= -1;
            }
            this.x += this.velocity[0];
            this.y += this.velocity[1];
            var d = new Date();
            var n = d.getTime();
            if (this.x > 400 || this.x < 0 || this.y > 400 || this.y < 0 || n > this.deathTime) {
                this.dead = true;
            }
        });
    }
    show() {
        strokeWeight(4);
        stroke(255);
        point(this.x, this.y)
    }
    update() {
        if (!this.dead) {
            this.move();
        }
        this.show();

    }
    getFitness(target) {
        if (this.dead) {
            this.fitness = 1 / dist(this.x, this.y, target.x, target.y);
            return this.fitness;
        }
        else {
            return undefined;
        }
    }
}