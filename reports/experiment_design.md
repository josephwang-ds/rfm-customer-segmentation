# Segment-Specific A/B Testing Plan

## Why Segment Before Testing

A global CRM experiment can hide treatment heterogeneity. High-value loyal customers, recent low-spend customers, and dormant customers may respond differently to the same promotion. RFM segmentation gives the experiment design a clean business hypothesis before randomization.

## Recommended Experiment Frames

### Dormant Low-Value Customers

- Hypothesis: a time-limited reactivation offer increases 14-day repeat purchase rate.
- Treatment: 20% discount or free-shipping email.
- Control: no offer or neutral reminder.
- Primary KPI: 14-day repeat purchase rate.
- Secondary KPI: revenue per customer.
- Guardrails: gross margin, cancellation rate, unsubscribe rate.

### Recent Low-Spend Customers

- Hypothesis: a second-purchase incentive increases conversion to repeat buyer.
- Treatment: personalized product recommendation plus small coupon.
- Control: generic product email.
- Primary KPI: second purchase within 30 days.
- Secondary KPI: AOV and category breadth.
- Guardrails: discount cost and return rate.

### Stable Repeat Customers

- Hypothesis: bundle recommendation increases AOV without hurting conversion.
- Treatment: cross-sell bundle recommendation.
- Control: best-seller recommendation.
- Primary KPI: revenue per recipient.
- Secondary KPI: AOV and units per order.
- Guardrails: conversion rate and return rate.

### High-Value Loyal Customers

- Hypothesis: VIP early access preserves loyalty without margin-eroding discounts.
- Treatment: early access or loyalty perk.
- Control: standard campaign.
- Primary KPI: revenue per customer.
- Secondary KPI: repeat purchase and retention.
- Guardrails: margin rate and customer support complaints.

## Experimental Discipline

- Randomize within each RFM segment.
- Keep a true holdout for incrementality.
- Do not reassign customers using post-treatment behavior.
- Pre-register primary KPI and guardrails.
- Compare treatment effect and cost-adjusted value by segment before rollout.
