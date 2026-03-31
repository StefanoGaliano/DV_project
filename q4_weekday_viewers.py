""" Q4 - Are the number of viewers for the episodes related to the weekday they were aired? 
    Two possible visualizations are:
    1. A violin plot showing the distribution of viewers for thursday vs sunday. (Two main days of airing)
    2. Smaller dot plot showing the number of viewers for each episode, colored by the remaining weekdays (monday, tuesday, wednesday, friday, saturday)."""

import pandas as pd
import altair as alt

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
violin_days = ["Thursday", "Sunday"]
non_violin_days = [day for day in day_order if day not in violin_days]

df = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")

base = alt.Chart(df, width=90, height=300)

# Violin only for Thu/Sun
violin = (
    base
    .transform_filter(alt.FieldOneOfPredicate(field="air_dayofweek", oneOf=violin_days))
    .transform_density(
        "us_viewers_in_millions",
        as_=["us_viewers_in_millions", "density"],
        extent=[0, df["us_viewers_in_millions"].max()],
        groupby=["air_dayofweek"],
    )
    .mark_area(orient="horizontal", opacity=0.7)
    .encode(
        x=alt.X("density:Q")
            .stack("center")
            .impute(None)
            .title(None)
            .axis(labels=False, values=[0],ticks=False, grid=False),
        y=alt.Y("us_viewers_in_millions:Q", title="US Viewers (Millions)"),
        color=alt.Color("air_dayofweek:N", legend=None),
    )
)
violin.save("outputs/q4_weekday_viewers_violin.html")

# Dot plot for all other days
dots = (
    base
    .transform_filter(alt.FieldOneOfPredicate(field="air_dayofweek", oneOf=non_violin_days))
    .transform_calculate(center="0")
    .mark_circle(size=60, opacity=0.9)
    .encode(
        x=alt.X("center:Q", title=None, axis=None),
        y=alt.Y("us_viewers_in_millions:Q", title="US Viewers (Millions)"),
        color=alt.Color("air_dayofweek:N", legend=None),
    )
)
dots.save("outputs/q4_weekday_viewers_dots.html")


combined = (
    alt.layer(violin, dots)
    .facet(
        column=alt.Column(
            "air_dayofweek:N",
            sort=day_order,
            header=alt.Header(titleOrient="bottom", labelOrient="bottom", labelPadding=0),
        )
    )
    .configure_view(stroke=None)
)

combined.save("outputs/q4_weekday_viewers_combined.html")

df = pd.read_csv("data/outputs/simpsons_episodes_clean.csv")

df_thursday_sunday = df[df["air_dayofweek"].isin(["Thursday", "Sunday"])]
#fill other days with NaN to avoid them being plotted in the violin plot
df_thursday_sunday = df_thursday_sunday.copy()
df_thursday_sunday.loc[~df_thursday_sunday["air_dayofweek"].isin(["Thursday", "Sunday"]), "us_viewers_in_millions"] = None


df_other_days = df[df["air_dayofweek"].isin(["Monday", "Tuesday", "Wednesday", "Friday", "Saturday"])]
# fill other days with NaN to avoid them being plotted in the dot plot
df_other_days = df_other_days.copy()
df_other_days.loc[~df_other_days["air_dayofweek"].isin(["Monday", "Tuesday", "Wednesday", "Friday", "Saturday"]), "us_viewers_in_millions"] = None

# Violin plot for Thursday vs Sunday
violin_alt = base.transform_filter(alt.FieldOneOfPredicate(field="air_dayofweek", oneOf=["Thursday", "Sunday"])).transform_density(
    'us_viewers_in_millions',
    as_=['us_viewers_in_millions', 'density'],
    extent=[0, df["us_viewers_in_millions"].max()],
    groupby=['air_dayofweek']
).mark_area(orient='horizontal').encode(
    alt.X('density:Q', title="Day of the Week", sort=["Thursday", "Sunday"])
        .stack('center')
        .impute(None)
        .title(None)
        .axis(labels=False, values=[0], grid=False, ticks=True),
    alt.Y('us_viewers_in_millions:Q', title="US Viewers (Millions)"),
    alt.Color('air_dayofweek:N', legend=None),
    alt.Column('air_dayofweek:N', sort=["Thursday", "Sunday"])
        .spacing(0)
        .header(titleOrient='bottom', labelOrient='bottom', labelPadding=0)
).configure_view(stroke=None)

violin_alt.save("outputs/q4_weekday_viewers_violion_alt.html")

# Dot Plot for remaining days
dot_alt = base.transform_filter(alt.FieldOneOfPredicate(field="air_dayofweek", oneOf=["Monday", "Tuesday", "Wednesday", "Friday", "Saturday"])).mark_circle(size=60).encode(
    x=alt.X('air_dayofweek:N', title="Day of the Week", sort=["Monday", "Tuesday", "Wednesday", "Friday", "Saturday"]),
    y=alt.Y('us_viewers_in_millions:Q', title="US Viewers (Millions)"),
    color=alt.Color('air_dayofweek:N', legend=None))

dot_alt.save("outputs/q4_weekday_viewers_dot_alt.html")

# combined_alt = alt.layer(violin_alt, dot_alt).configure_view(stroke=None).configure_axisX(labelAngle=0)
# combined_alt.save("outputs/q4_weekday_viewers_combined_alt.html")