"""Simple script to slice up given images for faster training."""

from input_producer import InputProducer

ip = InputProducer('data', 'png')

ip.create_sliced_folders()
