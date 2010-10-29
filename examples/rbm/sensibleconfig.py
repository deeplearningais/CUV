
"""
sensibleconfig
~~~~~~~~~~~~~~

So you have some configuration variables, and you want them to be available to
be in any number of ini-like files, as well as overridable from the environment,
and overridable from the command line. Define once, and use.


>>> options = [
...    Option('debug', 'Run in debug mode', False,
...           short_name='d', converter=bool, action='store_true'),
... ]
>>> conf = Config(options)
>>> conf.debug # Will start as the default value
False

This time we shall pass an env prefix to look up on, so as not to pollute any
environment namespace too badly:

>>> conf = Config(options, 'PONY')
>>> conf.grab_from_env({'PONY_DEBUG': '1'})
>>> conf.debug
True

.. note::
    The optional env dict parameter will default to os.environ if it is omitted.

Now we can grab some stuff from argv:

>>> conf = Config(options)
>>> conf.grab_from_argv(['--debug'])
[]
>>> conf.debug
True

Also, remember that you can serialize these things:

>>> conf = Config(options)
>>> conf.to_dict()
{'debug': False}

.. note::
    In real life you would import this::

        from sensibleconfig import Config, Option

So as you can see above, the options were declared, and then the config object
was created from those options. It is imagined that an application may collate
the options from many different places, such as plugins which wish to define
their own options.
"""

import os, optparse, ConfigParser


class Option(object):
    """An option definition.

    This should be declared before creating the Config instance and is used to
    tell the config how to manage the options it finds.

    :param name: a unique option name.
    :param help: a help text or description for the option.
    :param default: the default value
    :param short_name: a single letter representing the short name for short
                       options. If omitted will use the first letter of the
                       name parameter.
    :param converter: a Python callable to pass the value into to convert to a
                      useful Python value.
    :param optparse_kw: additional keyword arguments to pass to the optparse
                        add_option call.

    >>> o = Option('test', 'this is a test option', 'default')
    >>> o.long_opt
    '--test'
    >>> o.short_opt
    '-t'
    """

    def __init__(self, name, help, default, short_name=None, converter=unicode,
                       **optparse_kw):
        self.name = name
        self.short_name = short_name
        self.help = help
        self.default = converter(default)
        self.converter = converter
        self.optparse_kw = optparse_kw

    @property
    def long_opt(self):
        return '--%s' % self.name

    @property
    def short_opt(self):
        return '-%s' % self.short_name



class Config(object):
    """
    The configuration instance.

    This has the values set as attributes on itself.
    """

    def __init__(self, options, env_prefix=None, usage=None):
        self._env_prefix = env_prefix
        self._parser = optparse.OptionParser(usage=usage)
        self._options = {}
        for opt in options:
            self.add_option(opt)

    def get_serialization_obj(self):
        D = {}
        allowed = [ int, float, str, list ]
        for name, opt in self.__dict__.items():
            if opt.__class__ in allowed:
                D[name] = opt
        return D
    def __getitem__(self, v):
        return self.__dict__[v]
    def add_option(self, opt):
        """Add an option declaration.

        :param opt: an Option instance.
        """
        self._options[opt.name] = opt
        if opt.short_name != None:
            self._parser.add_option(opt.short_opt, opt.long_opt,
                                    default=opt.default,
                                    help=opt.help, **opt.optparse_kw)
        else:
            self._parser.add_option(opt.long_opt, default=opt.default,
                                    help=opt.help, **opt.optparse_kw)
        setattr(self, opt.name, opt.default)

    def grab_from_env(self, env=None):
        """
        Search the env dict for overriden values.

        This will employ the env_prefix, or if not set just capitalize the name
        of the variables.

        :param env: is an optional dict, if omitted, os.environ will be used.
        """
        if env is None:
            env = os.environ
        for name, opt in self._options.items():
            if self._env_prefix:
                env_name = '%s_%s' % (self._env_prefix, name.upper())
            else:
                env_name = name.upper()
            raw = env.get(env_name)
            if raw:
                val = opt.converter(raw)
                setattr(self, name, val)

    def grab_from_file(self, path):
        """
        Get values from an ini config file.
        """
        file = _ConfigFile(path)
        for name, opt in file.flat_options.items():
            if name in self._options:
                val = self._options[name].converter(opt)
                #print "Set ", name, " to ", val
                setattr(self, name, val)

    def grab_from_argv(self, argv):
        """
        Get values from argv list, and return the positional arguments.
        """
        opts, args = self._parser.parse_args(argv)

        for name, opt in self._options.items():
            raw = getattr(opts, name, None)
            if raw and raw != opt.default:
                val = opt.converter(raw)
                setattr(self, name, val)

        return args

    def grab_from_dict(self, d):
        """
        Load the options from a dict.

        This is useful if you want to dump or import options say as json.
        """
        for k, v in d.items():
            setattr(self, k, v)

    def items(self):
        for name in self._options:
            yield name, getattr(self, name)

    def __getattr__(self, name):
        if self.__dict__.has_key(name):
            return self.__dict__[name]
        #name = name.replace('-','_')
        if self.__dict__.has_key(name):
            return self.__dict__[name]
        raise AttributeError, name  # <<< DON'T FORGET THIS LINE !!

    def to_dict(self):
        """
        Serialize the options to a dict.

        This is useful if you want to dump or import options say as json.
        """
        return dict(self.items())


class _ConfigFile(ConfigParser.ConfigParser):
    """
    Flat config parser.

    Because we define defaults in a slightly different way, there is no reason
    to keep a strict hierarchy in the config file. So we flatten it.
    """

    def __init__(self, path):
        ConfigParser.ConfigParser.__init__(self)
        self.read(path)
        options = {}
        for section in self.sections():
            for name in self.options(section):
                if section == 'global':
                    options[name] = self.get(section, name)
                else:
                    options[section + '_' + name] = self.get(section, name)
        self.flat_options = options

