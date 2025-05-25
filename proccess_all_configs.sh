#!/bin/bash

for config in configs/*.yaml; do
    python3 results_generator.py "$config" &
done

wait