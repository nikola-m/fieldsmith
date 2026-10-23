"""Tiny helpers shared by the example scripts."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympy as sp


def section(title):
    print(f"\n=== {title} ===")


def show(label, value):
    print(f"{label}: {value}")


def ok(label, flag):
    print(f"[{'PASS' if flag else 'FAIL'}] {label}")
    return flag
