# generated from rosidl_generator_py/resource/_idl.py.em
# with input from interface_protocol:msg/Tts.idl
# generated code does not contain a copyright notice


# Import statements for member types

import builtins  # noqa: E402, I100

import rosidl_parser.definition  # noqa: E402, I100


class Metaclass_Tts(type):
    """Metaclass of message 'Tts'."""

    _CREATE_ROS_MESSAGE = None
    _CONVERT_FROM_PY = None
    _CONVERT_TO_PY = None
    _DESTROY_ROS_MESSAGE = None
    _TYPE_SUPPORT = None

    __constants = {
    }

    @classmethod
    def __import_type_support__(cls):
        try:
            from rosidl_generator_py import import_type_support
            module = import_type_support('interface_protocol')
        except ImportError:
            import logging
            import traceback
            logger = logging.getLogger(
                'interface_protocol.msg.Tts')
            logger.debug(
                'Failed to import needed modules for type support:\n' +
                traceback.format_exc())
        else:
            cls._CREATE_ROS_MESSAGE = module.create_ros_message_msg__msg__tts
            cls._CONVERT_FROM_PY = module.convert_from_py_msg__msg__tts
            cls._CONVERT_TO_PY = module.convert_to_py_msg__msg__tts
            cls._TYPE_SUPPORT = module.type_support_msg__msg__tts
            cls._DESTROY_ROS_MESSAGE = module.destroy_ros_message_msg__msg__tts

    @classmethod
    def __prepare__(cls, name, bases, **kwargs):
        # list constant names here so that they appear in the help text of
        # the message class under "Data and other attributes defined here:"
        # as well as populate each message instance
        return {
        }


class Tts(metaclass=Metaclass_Tts):
    """Message class 'Tts'."""

    __slots__ = [
        '_text',
        '_language',
        '_speaker',
        '_rate',
    ]

    _fields_and_field_types = {
        'text': 'string',
        'language': 'string',
        'speaker': 'string',
        'rate': 'int64',
    }

    SLOT_TYPES = (
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.UnboundedString(),  # noqa: E501
        rosidl_parser.definition.BasicType('int64'),  # noqa: E501
    )

    def __init__(self, **kwargs):
        assert all('_' + key in self.__slots__ for key in kwargs.keys()), \
            'Invalid arguments passed to constructor: %s' % \
            ', '.join(sorted(k for k in kwargs.keys() if '_' + k not in self.__slots__))
        self.text = kwargs.get('text', str())
        self.language = kwargs.get('language', str())
        self.speaker = kwargs.get('speaker', str())
        self.rate = kwargs.get('rate', int())

    def __repr__(self):
        typename = self.__class__.__module__.split('.')
        typename.pop()
        typename.append(self.__class__.__name__)
        args = []
        for s, t in zip(self.__slots__, self.SLOT_TYPES):
            field = getattr(self, s)
            fieldstr = repr(field)
            # We use Python array type for fields that can be directly stored
            # in them, and "normal" sequences for everything else.  If it is
            # a type that we store in an array, strip off the 'array' portion.
            if (
                isinstance(t, rosidl_parser.definition.AbstractSequence) and
                isinstance(t.value_type, rosidl_parser.definition.BasicType) and
                t.value_type.typename in ['float', 'double', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64']
            ):
                if len(field) == 0:
                    fieldstr = '[]'
                else:
                    assert fieldstr.startswith('array(')
                    prefix = "array('X', "
                    suffix = ')'
                    fieldstr = fieldstr[len(prefix):-len(suffix)]
            args.append(s[1:] + '=' + fieldstr)
        return '%s(%s)' % ('.'.join(typename), ', '.join(args))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.text != other.text:
            return False
        if self.language != other.language:
            return False
        if self.speaker != other.speaker:
            return False
        if self.rate != other.rate:
            return False
        return True

    @classmethod
    def get_fields_and_field_types(cls):
        from copy import copy
        return copy(cls._fields_and_field_types)

    @builtins.property
    def text(self):
        """Message field 'text'."""
        return self._text

    @text.setter
    def text(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'text' field must be of type 'str'"
        self._text = value

    @builtins.property
    def language(self):
        """Message field 'language'."""
        return self._language

    @language.setter
    def language(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'language' field must be of type 'str'"
        self._language = value

    @builtins.property
    def speaker(self):
        """Message field 'speaker'."""
        return self._speaker

    @speaker.setter
    def speaker(self, value):
        if __debug__:
            assert \
                isinstance(value, str), \
                "The 'speaker' field must be of type 'str'"
        self._speaker = value

    @builtins.property
    def rate(self):
        """Message field 'rate'."""
        return self._rate

    @rate.setter
    def rate(self, value):
        if __debug__:
            assert \
                isinstance(value, int), \
                "The 'rate' field must be of type 'int'"
            assert value >= -9223372036854775808 and value < 9223372036854775808, \
                "The 'rate' field must be an integer in [-9223372036854775808, 9223372036854775807]"
        self._rate = value
