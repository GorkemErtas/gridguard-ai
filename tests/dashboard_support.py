from gridguard.i18n import NAVIGATION, group_for_workspace


def app_file(*args, language="en", **kwargs):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(*args, **kwargs)
    app.session_state["language"] = language
    return app


def open_workspace(app, workspace):
    """Exercise the actual consolidated navigation, retaining existing EN assertions."""
    group = group_for_workspace(workspace)
    if app.sidebar.radio(key="section").value != group:
        if len(NAVIGATION[group]) > 1:
            app.session_state[f"view_{group}"] = workspace
        app.sidebar.radio(key="section").set_value(group).run()
    elif len(NAVIGATION[group]) > 1:
        app.segmented_control(key=f"view_{group}").set_value(workspace).run()
    else:
        app.run()
    return app
