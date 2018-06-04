let agents = new Array(100);
let newAgents = new Array(100);
const target = { x: 100, y: 300 };
function setup() {
    createCanvas(400, 400);
    background(0);
    for (let i = 0; i < agents.length; i++) {
        agents[i] = new Agent(200, 0);
    }
}

function draw() {
    background(0);
    point(target.x, target.y);
    let allDead = true;
    for (let i = 0; i < agents.length; i++) {
        agents[i].update();
        allDead = allDead && agents[i].dead;
    }
    if (allDead) {
        for (let i = 0; i < agents.length; i++) {
            agents[i].getFitness(target);
        }
        agents.sort((a, b) => {
            if (a.fitness > b.fitness) {
                return -1;
            } else {
                return 1;
            }
        });
        for (let i = 0; i < agents.length / 4; i++) {
            tf.tidy(() => {
                newAgents[i] = agents[i];
                newAgents[i].dead = false;
                newAgents[i].x = 200;
                newAgents[i].y = 0;
            });
        }
        for (let i = 25; i < agents.length; i++) {
            tf.tidy(() => {
                const weights = agents[floor(random(0, 10))].brain.getWeights();
                let a = weights[0].dataSync();
                let b = weights[2].dataSync();
                for (let j = 0; j < a.length; j++) {
                    if (random(1) < .1) {
                        a[j] = a[j] + randomGaussian() * 0.5;
                    }
                }
                for (let j = 0; j < b.length; j++) {
                    if (random(1) < .1) {
                        b[j] = b[j] + randomGaussian() * 0.5;
                    }
                }
                const c = new Array(4).fill(0).map(x => new Array(8));
                const d = new Array(8).fill(0).map(x => new Array(4));
                for (let j = 0; j < a.length; j++) {
                    c[floor(j / 8)][j % 8] = a[j];
                }
                for (let j = 0; j < b.length; j++) {
                    d[floor(j / 4)][j % 4] = b[j];
                }
                weights[0].assign(tf.tensor(c));
                weights[2].assign(tf.tensor(d));
                newAgents[i] = new Agent(200, 0, weights);
            });
        }
        tf.tidy(() => {
            agents = newAgents;
            newAgents = new Array(100);
        });
    }
    strokeWeight(10);
    stroke(255, 0, 0);
}