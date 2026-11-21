# Project 23: Online Shopping Delivery & Logistics Analysis (Python, NumPy, Pandas)
# Run from the folder that contains the data/ folder:  python project23_code.py


import pandas as pd
import numpy as np
from pathlib import Path

# Show all columns when printing tables
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------- Settings ----------
BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "23_delivery_performance.csv"
CLEAN_FILE = BASE_DIR / "cleaned_delivery_data.csv"
SAMPLE_SIZE = 678          # number of records to keep
RANDOM_SEED = 2026         # fixed seed so the same 678 rows are chosen every run

# Delay rules (my own, documented above)
ON_TIME_LIMIT = 5          # <= 5 days  -> On-Time
SLIGHT_LIMIT = 10          # 6-10 days  -> Slightly Delayed ; > 10 -> Highly Delayed
DELIVERED = "Delivered"

print("\n" + "=" * 70)
print("Task 1: Load the dataset and inspect shape, columns and data types")
print("=" * 70)

raw = pd.read_csv(RAW_FILE)
print("Shape of raw file (rows, columns):", raw.shape)
print("\nColumn names:", raw.columns.tolist())
print("\nData types:")
print(raw.dtypes)
print("\nFirst 5 rows:")
print(raw.head())

print("\n" + "=" * 70)
print("Task 2: Missing values and duplicate Shipment_ID records (raw file)")
print("=" * 70)

print("Missing values per column (raw file):")
print(raw.isnull().sum())

dup_rows = raw.duplicated(subset="Shipment_ID", keep=False).sum()
dup_extra = raw.duplicated(subset="Shipment_ID").sum()
print("\nRows that share a Shipment_ID with another row:", dup_rows)
print("Extra duplicate rows that will be removed:", dup_extra)
print("Are the duplicates exact copies of whole rows?", raw.duplicated().sum() == dup_extra)

# Quick consistency checks on the text columns (stray spaces / mixed case)
text_cols = ["Shipping_Method", "Delivery_Partner", "Delivery_Status", "Origin_City", "Destination_City"]
for col in text_cols:
    clean_unique = raw[col].str.strip().str.title().nunique()
    print(f"{col}: {raw[col].nunique()} unique values -> {clean_unique} after trimming/title-casing")

print("\n" + "=" * 70)
print("Building my working dataset")
print("=" * 70)
# 1. Remove duplicate `Shipment_ID` rows (keep the first copy).
# 2. Trim spaces in text columns.
# 3. Randomly keep **678** shipments (fixed seed). The rest are removed, as required by my brief.

df = raw.drop_duplicates(subset="Shipment_ID", keep="first").copy()
print("After removing duplicate IDs:", df.shape)

for col in text_cols:
    df[col] = df[col].str.strip()

df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).sort_values("Shipment_ID").reset_index(drop=True)
print("Working dataset kept:", df.shape)
print("Shipment_ID is unique:", df["Shipment_ID"].is_unique)

print("\nMissing values in the 678-row working dataset:")
print(df.isnull().sum())

print("\n" + "=" * 70)
print("Task 4 (done before Task 3): Convert dates to datetime")
print("=" * 70)
# Dates must be real datetime values before I can handle missing delivery dates and calculate durations.

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Delivery_Date"] = pd.to_datetime(df["Delivery_Date"], errors="coerce")

print("Unreadable Order_Date values:", df["Order_Date"].isna().sum())
print("Order dates range from", df["Order_Date"].min().date(), "to", df["Order_Date"].max().date())
print("Delivery dates before order dates:", (df["Delivery_Date"] < df["Order_Date"]).sum())
print(df[["Order_Date", "Delivery_Date"]].dtypes)

print("\n" + "=" * 70)
print("Task 3: Handle missing Delivery_Date and Shipping_Cost (justified method)")
print("=" * 70)
# **Delivery_Date**
# - Status = *Delivered* but date missing: the parcel did arrive, so the date was lost. I estimate the delivery time using the **median delivery days of the same shipping method** (median is not pulled by extreme values), and mark these rows with `Date_Imputed = True`.
# - Status = *In Transit / Returned / Failed* and date missing: a missing date is genuine, so it stays empty.
# **Shipping_Cost**
# - Cost depends on the service type and partner, so I fill with the **median cost of the same Shipping_Method + Delivery_Partner** group. Rows are marked with `Cost_Imputed = True`.

# ---- Delivery_Date ----
df["Date_Imputed"] = False
known = df[(df["Delivery_Status"] == DELIVERED) & df["Delivery_Date"].notna()].copy()
known["days"] = (known["Delivery_Date"] - known["Order_Date"]).dt.days
median_days_by_method = known.groupby("Shipping_Method")["days"].median()
print("Median delivery days by method (used for filling):")
print(median_days_by_method)

lost_date = (df["Delivery_Status"] == DELIVERED) & df["Delivery_Date"].isna()
print("\nDelivered shipments with a lost delivery date:", lost_date.sum())
fill_days = np.ceil(df.loc[lost_date, "Shipping_Method"].map(median_days_by_method))   # whole days
df.loc[lost_date, "Delivery_Date"] = df.loc[lost_date, "Order_Date"] + pd.to_timedelta(fill_days, unit="D")
df.loc[lost_date, "Date_Imputed"] = True

# ---- Shipping_Cost ----
df["Cost_Imputed"] = df["Shipping_Cost"].isna()
group_median = df.groupby(["Shipping_Method", "Delivery_Partner"])["Shipping_Cost"].transform("median")
df["Shipping_Cost"] = df["Shipping_Cost"].fillna(group_median)
df["Shipping_Cost"] = df["Shipping_Cost"].fillna(df["Shipping_Cost"].median())   # safety net
print("Shipping_Cost values filled:", df["Cost_Imputed"].sum())

print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("(Remaining empty Delivery_Date rows are genuine: shipments that are not delivered.)")
print(df.loc[df["Delivery_Date"].isna(), "Delivery_Status"].value_counts())

print("\n" + "=" * 70)
print("Task 5: Calculate Delivery_Days (and my extra helper columns)")
print("=" * 70)
# `Delivery_Days = Delivery_Date - Order_Date`. I also add `Order_Month`, `Order_YearMonth`, `Order_Weekday`, `Route` and `Cost_Per_Day`.

df["Delivery_Days"] = (df["Delivery_Date"] - df["Order_Date"]).dt.days

# Extra columns (my own additions)
df["Order_Year"] = df["Order_Date"].dt.year
df["Order_Month"] = df["Order_Date"].dt.month_name()
df["Order_YearMonth"] = df["Order_Date"].dt.to_period("M").astype(str)
df["Order_Weekday"] = df["Order_Date"].dt.day_name()
df["Route"] = df["Origin_City"] + " -> " + df["Destination_City"]
df["Same_City"] = df["Origin_City"] == df["Destination_City"]
df["Cost_Per_Day"] = (df["Shipping_Cost"] / df["Delivery_Days"]).round(2)

print(df[["Shipment_ID", "Order_Date", "Delivery_Date", "Delivery_Days", "Route", "Cost_Per_Day"]].head())
print("\nDelivery_Days summary (all rows that have a delivery date):")
print(df["Delivery_Days"].describe().round(2))

# Consistency check: statuses that still carry a delivery date
print("\nShipments with a delivery date, by status:")
print(df[df["Delivery_Date"].notna()]["Delivery_Status"].value_counts())

# Delivered shipments are used for all duration analysis (a failed/returned parcel has no real delivery time)
delivered_df = df[df["Delivery_Status"] == DELIVERED].copy()
print("\nDelivered shipments used for duration analysis:", len(delivered_df))

# Save the cleaned dataset
final_columns = ["Shipment_ID", "Order_Date", "Delivery_Date", "Shipping_Method", "Delivery_Partner", "Shipping_Cost",
                 "Delivery_Status", "Origin_City", "Destination_City", "Delivery_Days", "Order_Year",
                 "Order_Month", "Order_YearMonth", "Order_Weekday", "Route", "Same_City", "Cost_Per_Day",
                 "Date_Imputed", "Cost_Imputed"]
out = df[final_columns].copy()
out["Order_Date"] = out["Order_Date"].dt.strftime("%Y-%m-%d")
out["Delivery_Date"] = out["Delivery_Date"].dt.strftime("%Y-%m-%d")
out.to_csv(CLEAN_FILE, index=False)
print("Cleaned file saved:", CLEAN_FILE, "| rows:", len(out))

print("\n" + "=" * 70)
print("Task 6: Total shipments, delivered shipments and delivery success percentage")
print("=" * 70)

total_shipments = len(df)
delivered_shipments = (df["Delivery_Status"] == DELIVERED).sum()
success_pct = delivered_shipments / total_shipments * 100

print("Total shipments    :", total_shipments)
print("Delivered shipments:", delivered_shipments)
print(f"Delivery success   : {success_pct:.2f}%")
print("\nFull status breakdown (%):")
print((df["Delivery_Status"].value_counts(normalize=True) * 100).round(2))

print("\n" + "=" * 70)
print("Task 7: Average delivery duration for each shipping method")
print("=" * 70)

avg_by_method = (delivered_df.groupby("Shipping_Method")["Delivery_Days"]
                 .agg(Delivered_Shipments="count", Avg_Days="mean", Median_Days="median", Min_Days="min", Max_Days="max")
                 .round(2).sort_values("Avg_Days"))
print(avg_by_method)

print("\n" + "=" * 70)
print("Task 8: Rank delivery partners by number of shipments and average delivery duration")
print("=" * 70)

partner_summary = df.groupby("Delivery_Partner").agg(Total_Shipments=("Shipment_ID", "count"))
partner_summary["Avg_Delivery_Days"] = delivered_df.groupby("Delivery_Partner")["Delivery_Days"].mean().round(2)
partner_summary["Rank_By_Shipments"] = partner_summary["Total_Shipments"].rank(ascending=False, method="min").astype(int)
partner_summary["Rank_By_Speed"] = partner_summary["Avg_Delivery_Days"].rank(ascending=True, method="min").astype(int)
print(partner_summary.sort_values("Rank_By_Shipments"))
print("\n(Rank_By_Speed: 1 = fastest average delivery)")

print("\n" + "=" * 70)
print("Task 9: Delivery success rate across shipping methods")
print("=" * 70)

success_by_method = (df.assign(Is_Delivered=df["Delivery_Status"] == DELIVERED)
                     .groupby("Shipping_Method")
                     .agg(Shipments=("Shipment_ID", "count"), Delivered=("Is_Delivered", "sum"), Success_Rate_Pct=("Is_Delivered", "mean")))
success_by_method["Success_Rate_Pct"] = (success_by_method["Success_Rate_Pct"] * 100).round(2)
print(success_by_method.sort_values("Success_Rate_Pct", ascending=False))

print("\n" + "=" * 70)
print("Task 10: The 10 shipments with the longest delivery duration")
print("=" * 70)
# Many shipments tie at the maximum number of days, so ties are broken by the **earliest order date**.

longest_10 = (delivered_df.sort_values(["Delivery_Days", "Order_Date"], ascending=[False, True])
              .head(10)[["Shipment_ID", "Delivery_Partner", "Shipping_Method", "Route", "Order_Date", "Delivery_Days"]])
print(longest_10.to_string(index=False))

print("\n" + "=" * 70)
print("Task 11: Dictionary of delivery counts by status, and a set of unique delivery partners")
print("=" * 70)

status_counts = df["Delivery_Status"].value_counts().to_dict()
unique_partners = set(df["Delivery_Partner"])

print("Delivery counts by status (dictionary):")
for status, count in status_counts.items():
    print(f"  {status}: {count}")
print("\nUnique delivery partners (set):", sorted(unique_partners))
print("Number of unique partners:", len(unique_partners))

print("\n" + "=" * 70)
print("Task 12: Function to label each shipment")
print("=" * 70)
# My stated limits: **On-Time** = up to 5 days, **Slightly Delayed** = 6 to 10 days, **Highly Delayed** = 11 days or more. Shipments that are not delivered get the label *Not Delivered*.

def label_delivery(days, status="Delivered", on_time_limit=ON_TIME_LIMIT, slight_limit=SLIGHT_LIMIT):
    """Return the delay label for one shipment using the stated day limits."""
    if status != "Delivered" or pd.isna(days):
        return "Not Delivered"
    if days <= on_time_limit:
        return "On-Time"
    elif days <= slight_limit:
        return "Slightly Delayed"
    else:
        return "Highly Delayed"

# Quick test of the function
print("Test 3 days  ->", label_delivery(3))
print("Test 8 days  ->", label_delivery(8))
print("Test 13 days ->", label_delivery(13))
print("Test failed  ->", label_delivery(4, status="Failed"))

df["Delay_Label"] = [label_delivery(d, s) for d, s in zip(df["Delivery_Days"], df["Delivery_Status"])]
delivered_df["Delay_Label"] = df.loc[delivered_df.index, "Delay_Label"]

print("\nDelay label counts (all shipments):")
print(df["Delay_Label"].value_counts())
print("\nDelay label share among delivered shipments (%):")
print((delivered_df["Delay_Label"].value_counts(normalize=True) * 100).round(2))

print("\n" + "=" * 70)
print("Task 13: NumPy statistics of Delivery_Days (delivered shipments)")
print("=" * 70)

days_array = delivered_df["Delivery_Days"].to_numpy(dtype=float)

print("Number of values :", days_array.size)
print("Mean             :", round(np.mean(days_array), 2))
print("Median           :", np.median(days_array))
print("Std deviation    :", round(np.std(days_array), 2))
print("Variance         :", round(np.var(days_array), 2))
print("Minimum / Maximum:", days_array.min(), "/", days_array.max())
print("90th percentile  :", np.percentile(days_array, 90))

print("\n" + "=" * 70)
print("Task 14: Unusually long deliveries (IQR method) and monthly performance")
print("=" * 70)

q1 = np.percentile(days_array, 25)
q3 = np.percentile(days_array, 75)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr

print(f"Q1 = {q1}, Q3 = {q3}, IQR = {iqr}")
print(f"Lower limit = {lower_limit}, Upper limit = {upper_limit}")

outliers = delivered_df[(delivered_df["Delivery_Days"] > upper_limit) | (delivered_df["Delivery_Days"] < lower_limit)]
print("\nNumber of IQR outliers:", len(outliers))
if len(outliers) == 0:
    print("No statistical outliers: the longest delivery (", int(days_array.max()),
          "days ) is inside the upper limit, so delivery times are spread evenly between",
          int(days_array.min()), "and", int(days_array.max()), "days.")
else:
    print(outliers[["Shipment_ID", "Delivery_Partner", "Delivery_Days"]].head(10))

# Monthly performance (by order Year-Month)
monthly = df.groupby("Order_YearMonth").agg(Shipments=("Shipment_ID", "count"),
                                              Delivered=("Delivery_Status", lambda s: (s == DELIVERED).sum()))
monthly["Success_Rate_Pct"] = (monthly["Delivered"] / monthly["Shipments"] * 100).round(2)
monthly["Avg_Delivery_Days"] = delivered_df.groupby("Order_YearMonth")["Delivery_Days"].mean().round(2)
monthly["Highly_Delayed_Pct"] = (delivered_df.groupby("Order_YearMonth")["Delay_Label"]
                                 .apply(lambda s: (s == "Highly Delayed").mean() * 100).round(2))
print(monthly.to_string())

# Months with at least 20 shipments are compared fairly
enough = monthly[monthly["Shipments"] >= 20]
best_month = enough["Avg_Delivery_Days"].idxmin()
worst_month = enough["Avg_Delivery_Days"].idxmax()
print(f"\nAmong months with 20+ shipments: fastest = {best_month} ({enough.loc[best_month, 'Avg_Delivery_Days']} days), "
      f"slowest = {worst_month} ({enough.loc[worst_month, 'Avg_Delivery_Days']} days)")

print("\n" + "=" * 70)
print("Extra analysis (my own additions)")
print("=" * 70)
# These go beyond the required tasks: partner delay percentages, busiest routes, cost vs. duration association and stale *In Transit* shipments.

# 1. Partner performance in more detail
partner_extra = df.assign(Is_Delivered=df["Delivery_Status"] == DELIVERED).groupby("Delivery_Partner").agg(
    Shipments=("Shipment_ID", "count"), Success_Rate_Pct=("Is_Delivered", "mean"))
partner_extra["Success_Rate_Pct"] = (partner_extra["Success_Rate_Pct"] * 100).round(2)
partner_extra["Highly_Delayed_Pct"] = (delivered_df.groupby("Delivery_Partner")["Delay_Label"]
                                       .apply(lambda s: (s == "Highly Delayed").mean() * 100).round(2))
partner_extra["Avg_Cost"] = df.groupby("Delivery_Partner")["Shipping_Cost"].mean().round(2)
print("Partner detail:")
print(partner_extra.sort_values("Success_Rate_Pct", ascending=False))

# 2. Busiest routes
print("\nTop 5 busiest routes:")
print(df["Route"].value_counts().head(5))

# 3. Same-city vs different-city delivery time
print("\nAverage delivery days, same-city vs different-city (delivered):")
print(delivered_df.groupby("Same_City")["Delivery_Days"].agg(["count", "mean"]).round(2))

# 4. Association between cost and delivery days (association only, not cause)
corr = np.corrcoef(delivered_df["Shipping_Cost"], delivered_df["Delivery_Days"])[0, 1]
print(f"\nCorrelation between Shipping_Cost and Delivery_Days: {corr:.3f}")
print("(A value near 0 means the two do not move together in this data. Correlation does not prove cause.)")

# 5. Stale In Transit shipments (relative to the latest order date in the data)
latest_date = df["Order_Date"].max()
in_transit = df[df["Delivery_Status"] == "In Transit"]
stale_in_transit = in_transit[(latest_date - in_transit["Order_Date"]).dt.days > 30]
print(f"\nShipments marked In Transit: {len(in_transit)}")
print(f"Of these, ordered more than 30 days before the latest order date ({latest_date.date()}): {len(stale_in_transit)}")

print("\n" + "=" * 70)
print("Lambda functions (extra)")
print("=" * 70)
# A lambda is a small one-line function without a name.

# 1. Percentage of a part in a whole
percent = lambda part, whole: round(part / whole * 100, 2)
print("Delivered share of all shipments (%):", percent(delivered_shipments, total_shipments))

# 2. Shipping-cost band using a lambda inside apply()
cost_band = lambda c: "Low" if c < 200 else ("Medium" if c < 400 else "High")
df["Cost_Band"] = df["Shipping_Cost"].apply(cost_band)
print("\nShipments by cost band (Low < 200, Medium 200-399, High 400+):")
print(df["Cost_Band"].value_counts())

# 3. Sort the status dictionary by count (key = lambda)
sorted_status = sorted(status_counts.items(), key=lambda item: item[1], reverse=True)
print("\nStatus counts sorted from highest to lowest:", sorted_status)

# 4. map() with a lambda: first 5 delivery durations converted from days to hours
hours = list(map(lambda d: int(d * 24), days_array[:5]))
print("\nFirst 5 delivery durations in hours:", hours)

# 5. filter() with a lambda: delivered shipments taking more than the Slightly Delayed limit
very_long = list(filter(lambda d: d > SLIGHT_LIMIT, days_array))
print(f"Delivered shipments taking more than {SLIGHT_LIMIT} days:", len(very_long))

# 6. Lambda inside agg(): percentage of delivered shipments taking more than 10 days, for each shipping method
long_share_by_method = delivered_df.groupby("Shipping_Method")["Delivery_Days"].agg(lambda s: round((s > SLIGHT_LIMIT).mean() * 100, 2))
print(f"\nPercentage of delivered shipments taking more than {SLIGHT_LIMIT} days, by shipping method:")
print(long_share_by_method)

print("\n" + "=" * 70)
print("Task 15: Shipment class")
print("=" * 70)
# A simple class with shipment ID, partner and delivery-days attributes (plus a method that reuses my labelling function).

class Shipment:
    """Represents one shipment."""
    def __init__(self, shipment_id, partner, delivery_days):
        self.shipment_id = shipment_id
        self.partner = partner
        self.delivery_days = delivery_days

    def delay_label(self):
        return label_delivery(self.delivery_days)

    def describe(self):
        return f"{self.shipment_id} | {self.partner} | {self.delivery_days:.0f} days | {self.delay_label()}"

print("The 3 slowest delivered shipments as Shipment objects:")
for _, row in longest_10.head(3).iterrows():
    print("  ", Shipment(row["Shipment_ID"], row["Delivery_Partner"], row["Delivery_Days"]).describe())

# One more simple class: summary of a delivery partner (OOP)
class DeliveryPartner:
    """Holds the summary of one delivery partner."""
    def __init__(self, name, shipments, avg_days):
        self.name = name
        self.shipments = shipments
        self.avg_days = avg_days

    def report(self):
        return f"{self.name}: {self.shipments} shipments, average {self.avg_days} days"

print("\nDelivery partners as DeliveryPartner objects (fastest first):")
for name, row in partner_summary.sort_values("Rank_By_Speed").iterrows():
    print("  ", DeliveryPartner(name, int(row["Total_Shipments"]), row["Avg_Delivery_Days"]).report())

print("\n" + "=" * 70)
print("Five recommendations (the numbers are calculated from the data above, not typed in)")
print("=" * 70)

fast_partner = partner_summary["Avg_Delivery_Days"].idxmin()
slow_partner = partner_summary["Avg_Delivery_Days"].idxmax()
best_method = success_by_method["Success_Rate_Pct"].idxmax()
weak_method = success_by_method["Success_Rate_Pct"].idxmin()
highly_delayed_pct = (delivered_df["Delay_Label"] == "Highly Delayed").mean() * 100
not_delivered_pct = 100 - success_pct
returned_failed = df["Delivery_Status"].isin(["Returned", "Failed"]).mean() * 100

recommendations = [
    f"1. Partner review: {slow_partner} has the slowest average delivery ({partner_summary.loc[slow_partner, 'Avg_Delivery_Days']} days) "
    f"and {fast_partner} the fastest ({partner_summary.loc[fast_partner, 'Avg_Delivery_Days']} days). Set a target delivery time in the contract "
    f"and review the partners every month.",

    f"2. Shipping method success: {weak_method} has the lowest success rate ({success_by_method.loc[weak_method, 'Success_Rate_Pct']}%) "
    f"compared with {best_method} ({success_by_method.loc[best_method, 'Success_Rate_Pct']}%). Record the reason for every failed or returned "
    f"{weak_method} shipment to find the main cause.",

    f"3. Delay control: {highly_delayed_pct:.1f}% of delivered shipments took {SLIGHT_LIMIT + 1} days or more. Create an alert for any open "
    f"shipment that crosses {ON_TIME_LIMIT} days so the team can act before it becomes Highly Delayed.",

    f"4. Tracking quality: {len(stale_in_transit)} shipments are still marked In Transit more than 30 days after ordering. Automatically flag "
    f"such records for checking, because they distort the {success_pct:.1f}% success rate.",

    f"5. Reducing unsuccessful deliveries: {not_delivered_pct:.1f}% of shipments are not delivered ({returned_failed:.1f}% are Returned or Failed). "
    f"Confirm addresses and delivery slots before dispatch. Month-wise, {worst_month} was slowest ({enough.loc[worst_month, 'Avg_Delivery_Days']} days) "
    f"and {best_month} fastest ({enough.loc[best_month, 'Avg_Delivery_Days']} days), so check what staffing or volume differed in the slow month.",
]
for r in recommendations:
    print(r)
    print()

print("Limitations: this is a synthetic dataset and differences between groups are small, so these are directions to investigate, not proven causes.")

